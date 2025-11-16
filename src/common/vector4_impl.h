#pragma once
#include "bfloat16.h"
#include "complex.h"
#include "defines.h"
#include "exception.h"
#include "half_fp16.h"
#include "mpp_defs.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include "safeCast.h"
#include "staticCast.h"
#include "vector_typetraits.h"
#include "vector2.h"
#include "vector3.h"
#include "vector4.h"
#include "vector4A.h"
#include <cmath>
#include <common/utilities.h>
#include <concepts>
#include <cstring>
#include <iostream>
#include <type_traits>

#ifdef IS_CUDA_COMPILER
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#else
namespace mpp
{
// these types are only used with CUDA, but nevertheless they need
// to be defined, so we set them to some known type of same size:
using nv_bfloat162 = int;
using half2        = float;
using float2       = double;
} // namespace mpp

// no arguments to these intrinsics directly depend on a template parameter,
// so a declaration must be available:
mpp::float2 __half22float2(mpp::half2);               // NOLINT
mpp::half2 __float22half2_rn(mpp::float2);            // NOLINT
mpp::float2 __bfloat1622float2(mpp::nv_bfloat162);    // NOLINT
mpp::nv_bfloat162 __float22bfloat162_rn(mpp::float2); // NOLINT

mpp::uint __vnegss2(mpp::uint);                                  // NOLINT
mpp::nv_bfloat162 __hneg2(mpp::nv_bfloat162);                    // NOLINT
mpp::half2 __hneg2(mpp::half2);                                  // NOLINT
mpp::uint __vabsss2(mpp::uint);                                  // NOLINT
mpp::uint __vaddus2(mpp::uint, mpp::uint);                       // NOLINT
mpp::uint __vaddss2(mpp::uint, mpp::uint);                       // NOLINT
mpp::uint __vsubus2(mpp::uint, mpp::uint);                       // NOLINT
mpp::uint __vsubss2(mpp::uint, mpp::uint);                       // NOLINT
mpp::uint __vabsdiffs2(mpp::uint, mpp::uint);                    // NOLINT
mpp::uint __vabsdiffu2(mpp::uint, mpp::uint);                    // NOLINT
mpp::uint __vmins2(mpp::uint, mpp::uint);                        // NOLINT
mpp::uint __vminu2(mpp::uint, mpp::uint);                        // NOLINT
mpp::uint __vmaxs2(mpp::uint, mpp::uint);                        // NOLINT
mpp::uint __vmaxu2(mpp::uint, mpp::uint);                        // NOLINT
mpp::uint __vcmpeq2(mpp::uint, mpp::uint);                       // NOLINT
mpp::uint __vcmpges2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmpgeu2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmpgts2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmpgtu2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmples2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmpleu2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmplts2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmpltu2(mpp::uint, mpp::uint);                      // NOLINT
mpp::uint __vcmpne2(mpp::uint, mpp::uint);                       // NOLINT
mpp::nv_bfloat162 __hadd2(mpp::nv_bfloat162, mpp::nv_bfloat162); // NOLINT
mpp::half2 __hadd2(mpp::half2, mpp::half2);                      // NOLINT
mpp::nv_bfloat162 __hsub2(mpp::nv_bfloat162, mpp::nv_bfloat162); // NOLINT
mpp::half2 __hsub2(mpp::half2, mpp::half2);                      // NOLINT
mpp::nv_bfloat162 __hmul2(mpp::nv_bfloat162, mpp::nv_bfloat162); // NOLINT
mpp::half2 __hmul2(mpp::half2, mpp::half2);                      // NOLINT
mpp::nv_bfloat162 __h2div(mpp::nv_bfloat162, mpp::nv_bfloat162); // NOLINT
mpp::half2 __h2div(mpp::half2, mpp::half2);                      // NOLINT
mpp::nv_bfloat162 h2exp(mpp::nv_bfloat162);                      // NOLINT
mpp::half2 h2exp(mpp::half2);                                    // NOLINT
mpp::nv_bfloat162 h2log(mpp::nv_bfloat162);                      // NOLINT
mpp::half2 h2log(mpp::half2);                                    // NOLINT
mpp::nv_bfloat162 h2sqrt(mpp::nv_bfloat162);                     // NOLINT
mpp::half2 h2sqrt(mpp::half2);                                   // NOLINT
mpp::nv_bfloat162 __habs2(mpp::nv_bfloat162);                    // NOLINT
mpp::half2 __habs2(mpp::half2);                                  // NOLINT
mpp::nv_bfloat162 __hmin2(mpp::nv_bfloat162, mpp::nv_bfloat162); // NOLINT
mpp::half2 __hmin2(mpp::half2, mpp::half2);                      // NOLINT
mpp::nv_bfloat162 __hmax2(mpp::nv_bfloat162, mpp::nv_bfloat162); // NOLINT
mpp::half2 __hmax2(mpp::half2, mpp::half2);                      // NOLINT
mpp::nv_bfloat162 h2floor(mpp::nv_bfloat162);                    // NOLINT
mpp::half2 h2floor(mpp::half2);                                  // NOLINT
mpp::nv_bfloat162 h2ceil(mpp::nv_bfloat162);                     // NOLINT
mpp::half2 h2ceil(mpp::half2);                                   // NOLINT
mpp::nv_bfloat162 h2rint(mpp::nv_bfloat162);                     // NOLINT
mpp::half2 h2rint(mpp::half2);                                   // NOLINT
mpp::nv_bfloat162 h2trunc(mpp::nv_bfloat162);                    // NOLINT
mpp::half2 h2trunc(mpp::half2);                                  // NOLINT
mpp::uint __heq2_mask(mpp::nv_bfloat162, mpp::nv_bfloat162);     // NOLINT
mpp::uint __heq2_mask(mpp::half2, mpp::half2);                   // NOLINT
mpp::uint __hgt2_mask(mpp::nv_bfloat162, mpp::nv_bfloat162);     // NOLINT
mpp::uint __hgt2_mask(mpp::half2, mpp::half2);                   // NOLINT
mpp::uint __hge2_mask(mpp::nv_bfloat162, mpp::nv_bfloat162);     // NOLINT
mpp::uint __hge2_mask(mpp::half2, mpp::half2);                   // NOLINT
mpp::uint __hlt2_mask(mpp::nv_bfloat162, mpp::nv_bfloat162);     // NOLINT
mpp::uint __hlt2_mask(mpp::half2, mpp::half2);                   // NOLINT
mpp::uint __hle2_mask(mpp::nv_bfloat162, mpp::nv_bfloat162);     // NOLINT
mpp::uint __hle2_mask(mpp::half2, mpp::half2);                   // NOLINT
mpp::uint __hne2_mask(mpp::nv_bfloat162, mpp::nv_bfloat162);     // NOLINT
mpp::uint __hne2_mask(mpp::half2, mpp::half2);                   // NOLINT
bool __hbeq2(mpp::nv_bfloat162, mpp::nv_bfloat162);              // NOLINT
bool __hbeq2(mpp::half2, mpp::half2);                            // NOLINT
bool __hbgt2(mpp::nv_bfloat162, mpp::nv_bfloat162);              // NOLINT
bool __hbgt2(mpp::half2, mpp::half2);                            // NOLINT
bool __hbge2(mpp::nv_bfloat162, mpp::nv_bfloat162);              // NOLINT
bool __hbge2(mpp::half2, mpp::half2);                            // NOLINT
bool __hblt2(mpp::nv_bfloat162, mpp::nv_bfloat162);              // NOLINT
bool __hblt2(mpp::half2, mpp::half2);                            // NOLINT
bool __hble2(mpp::nv_bfloat162, mpp::nv_bfloat162);              // NOLINT
bool __hble2(mpp::half2, mpp::half2);                            // NOLINT
bool __hbne2(mpp::nv_bfloat162, mpp::nv_bfloat162);              // NOLINT
#endif

namespace mpp
{

#pragma region Constructors

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::FromUint(const uint &aUint) noexcept
    requires ByteSizeType<T>
{
    return Vector4(*reinterpret_cast<const Vector4<T> *>(&aUint));
}

/// <summary>
/// Usefull constructor for SIMD instructions (performs the bitshifts needed to merge compare results for 2 byte
/// types)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::FromUint(uint aUintLO, uint aUintHI) noexcept
    requires IsByte<T>
{
    // from two UInts building an ULong of value 0xaUintHIaUintLO or 0xDDDDCCCCBBBBAAAA, shift bits and mask them so
    // that we get an UInt 0xDDCCBBAA
    aUintLO = (aUintLO & 0xFF) | ((aUintLO >> 8) & 0xFF00);           // NOLINT
    aUintHI = (aUintHI & 0xFF000000) | ((aUintHI << 8) & 0x00FF0000); // NOLINT
    aUintLO |= aUintHI;                                               // NOLINT

    // avoid GCC warning: "dereferencing type-punned pointer will break strict-aliasing rules"
    // by using a union
    const union
    {
        uint UInt;
        Vector4<T> Vec4;
    } ret{aUintLO};
    return ret.Vec4; // NOLINT(cppcoreguidelines-pro-type-union-access)
    // return Vector4(*reinterpret_cast<const Vector4<T> *>(&aUintLO));
}

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::FromUlong(const ulong64 &aUlong) noexcept
    requires TwoBytesSizeType<T>
{
    return Vector4(*reinterpret_cast<const Vector4<T> *>(&aUlong));
}

/// <summary>
/// Type conversion with saturation if needed<para/>
/// E.g.: when converting int to byte, values are clamped to 0..255<para/>
/// But when converting byte to int, no clamping operation is performed.
/// </summary>
template <Number T> template <Number T2> DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec) noexcept
{
    if constexpr (need_saturation_clamp_v<T2, T>)
    {
        Vector4<T2> temp(aVec);
        temp.template ClampToTargetType<T>();
        x = StaticCast<T2, T>(temp.x);
        y = StaticCast<T2, T>(temp.y);
        z = StaticCast<T2, T>(temp.z);
        w = StaticCast<T2, T>(temp.w);
    }
    else
    {
        x = StaticCast<T2, T>(aVec.x);
        y = StaticCast<T2, T>(aVec.y);
        z = StaticCast<T2, T>(aVec.z);
        w = StaticCast<T2, T>(aVec.w);
    }
}

/// <summary>
/// Type conversion with saturation if needed<para/>
/// E.g.: when converting int to byte, values are clamped to 0..255<para/>
/// But when converting byte to int, no clamping operation is performed.<para/>
/// If we can modify the input variable, no need to allocate temporary storage for clamping.
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(Vector4<T2> &aVec) noexcept
    // Disable the non-const variant for half and bfloat to / from float,
    // otherwise the const specialization will never be picked up:
    requires(!(IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>) &&
             !(IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>)) &&
            (!std::same_as<T, T2>)
{
    if constexpr (need_saturation_clamp_v<T2, T>)
    {
        aVec.template ClampToTargetType<T>();
    }
    x = StaticCast<T2, T>(aVec.x);
    y = StaticCast<T2, T>(aVec.y);
    z = StaticCast<T2, T>(aVec.z);
    w = StaticCast<T2, T>(aVec.w);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to BFloat2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec) noexcept
    requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>
{
    const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = __float22bfloat162_rn(aVecPtr[0]);
    thisPtr[1]            = __float22bfloat162_rn(aVecPtr[1]);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to BFloat2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec, RoundingMode aRoundingMode)
    requires IsBFloat16<T> && IsFloat<T2>
{
    if constexpr (CUDA_ONLY<T>)
    {
        if (aRoundingMode == RoundingMode::NearestTiesToEven)
        {
            const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
            nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
            thisPtr[0]            = __float22bfloat162_rn(aVecPtr[0]);
            thisPtr[1]            = __float22bfloat162_rn(aVecPtr[1]);
        }
        else
        {
            x = BFloat16(aVec.x, aRoundingMode);
            y = BFloat16(aVec.y, aRoundingMode);
            z = BFloat16(aVec.z, aRoundingMode);
            w = BFloat16(aVec.w, aRoundingMode);
        }
    }
    else
    {
        x = BFloat16(aVec.x, aRoundingMode);
        y = BFloat16(aVec.y, aRoundingMode);
        z = BFloat16(aVec.z, aRoundingMode);
        w = BFloat16(aVec.w, aRoundingMode);
    }
}

/// <summary>
/// Type conversion using CUDA intrinsics for BFloat2 to float2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec) noexcept
    requires IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>
{
    const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
    float2 *thisPtr             = reinterpret_cast<float2 *>(this);
    thisPtr[0]                  = __bfloat1622float2(aVecPtr[0]);
    thisPtr[1]                  = __bfloat1622float2(aVecPtr[1]);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to half2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec) noexcept
    requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>
{
    const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    thisPtr[0]            = __float22half2_rn(aVecPtr[0]);
    thisPtr[1]            = __float22half2_rn(aVecPtr[1]);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to half2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec, RoundingMode aRoundingMode)
    requires IsHalfFp16<T> && IsFloat<T2>
{
    if constexpr (CUDA_ONLY<T>)
    {
        if (aRoundingMode == RoundingMode::NearestTiesToEven)
        {
            const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
            half2 *thisPtr        = reinterpret_cast<half2 *>(this);
            thisPtr[0]            = __float22half2_rn(aVecPtr[0]);
            thisPtr[1]            = __float22half2_rn(aVecPtr[1]);
        }
        else
        {
            x = HalfFp16(aVec.x, aRoundingMode);
            y = HalfFp16(aVec.y, aRoundingMode);
            z = HalfFp16(aVec.z, aRoundingMode);
            w = HalfFp16(aVec.w, aRoundingMode);
        }
    }
    else
    {
        x = HalfFp16(aVec.x, aRoundingMode);
        y = HalfFp16(aVec.y, aRoundingMode);
        z = HalfFp16(aVec.z, aRoundingMode);
        w = HalfFp16(aVec.w, aRoundingMode);
    }
}

/// <summary>
/// Type conversion for complex with rounding (only for float to bfloat/halffloat)
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec, RoundingMode aRoundingMode)
    requires ComplexFloatingPoint<T> && ComplexFloatingPoint<T2> &&
                 NonNativeFloatingPoint<complex_basetype_t<remove_vector_t<T>>> &&
                 std::same_as<float, complex_basetype_t<remove_vector_t<T2>>>
    : x(T(aVec.x, aRoundingMode)), y(T(aVec.y, aRoundingMode)), z(T(aVec.z, aRoundingMode)), w(T(aVec.w, aRoundingMode))
{
}

/// <summary>
/// Type conversion using CUDA intrinsics for half2 to float2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector4<T>::Vector4(const Vector4<T2> &aVec) noexcept
    requires IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>
{
    const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
    float2 *thisPtr      = reinterpret_cast<float2 *>(this);
    thisPtr[0]           = __half22float2(aVecPtr[0]);
    thisPtr[1]           = __half22float2(aVecPtr[1]);
}

/// <summary>
/// converter to uint for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T>::operator const uint &() const
    requires ByteSizeType<T>
{
    return *reinterpret_cast<const uint *>(this);
}

/// <summary>
/// converter to uint for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T>::operator uint &()
    requires ByteSizeType<T>
{
    return *reinterpret_cast<uint *>(this);
}

#pragma endregion
#pragma region Operators
/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::EqEps(const Vector4<T> &aLeft, const Vector4<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && HostCode<T>
{
    Vector4<T> left  = aLeft;
    Vector4<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);
    MakeNANandINFValid(left.z, right.z);
    MakeNANandINFValid(left.w, right.w);

    bool res = std::abs(left.x - right.x) <= aEpsilon;
    res &= std::abs(left.y - right.y) <= aEpsilon;
    res &= std::abs(left.z - right.z) <= aEpsilon;
    res &= std::abs(left.w - right.w) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::EqEps(const Vector4<T> &aLeft, const Vector4<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && DeviceCode<T>
{
    Vector4<T> left  = aLeft;
    Vector4<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);
    MakeNANandINFValid(left.z, right.z);
    MakeNANandINFValid(left.w, right.w);

    bool res = abs(left.x - right.x) <= aEpsilon;
    res &= abs(left.y - right.y) <= aEpsilon;
    res &= abs(left.z - right.z) <= aEpsilon;
    res &= abs(left.w - right.w) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::EqEps(const Vector4<T> &aLeft, const Vector4<T> &aRight, T aEpsilon)
    requires Is16BitFloat<T>
{
    Vector4<T> left  = aLeft;
    Vector4<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);
    MakeNANandINFValid(left.z, right.z);
    MakeNANandINFValid(left.w, right.w);

    bool res = T::Abs(left.x - right.x) <= aEpsilon;
    res &= T::Abs(left.y - right.y) <= aEpsilon;
    res &= T::Abs(left.z - right.z) <= aEpsilon;
    res &= T::Abs(left.w - right.w) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::EqEps(const Vector4<T> &aLeft, const Vector4<T> &aRight, complex_basetype_t<T> aEpsilon)
    requires ComplexFloatingPoint<T>
{
    bool res = T::EqEps(aLeft.x, aRight.x, aEpsilon);
    res &= T::EqEps(aLeft.y, aRight.y, aEpsilon);
    res &= T::EqEps(aLeft.z, aRight.z, aEpsilon);
    res &= T::EqEps(aLeft.w, aRight.w, aEpsilon);
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::operator<(const Vector4<T> &aOther) const
    requires RealNumber<T>
{
    bool res = x < aOther.x;
    res &= y < aOther.y;
    res &= z < aOther.z;
    res &= w < aOther.w;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::operator<=(const Vector4<T> &aOther) const
    requires RealNumber<T>
{
    bool res = x <= aOther.x;
    res &= y <= aOther.y;
    res &= z <= aOther.z;
    res &= w <= aOther.w;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::operator>(const Vector4<T> &aOther) const
    requires RealNumber<T>
{
    bool res = x > aOther.x;
    res &= y > aOther.y;
    res &= z > aOther.z;
    res &= w > aOther.w;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector4<T>::operator>=(const Vector4<T> &aOther) const
    requires RealNumber<T>
{
    bool res = x >= aOther.x;
    res &= y >= aOther.y;
    res &= z >= aOther.z;
    res &= w >= aOther.w;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T> DEVICE_CODE bool Vector4<T>::operator==(const Vector4<T> &aOther) const
{
    bool res = x == aOther.x;
    res &= y == aOther.y;
    res &= z == aOther.z;
    res &= w == aOther.w;
    return res;
}

/// <summary>
/// Returns true if any element comparison is true
/// </summary>
template <Number T> DEVICE_CODE bool Vector4<T>::operator!=(const Vector4<T> &aOther) const
{
    bool res = x != aOther.x;
    res |= y != aOther.y;
    res |= z != aOther.z;
    res |= w != aOther.w;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator==(const Vector4<T> &aOther) const
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpeq4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>=(const Vector4<T> &aOther) const
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpgeu4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>(const Vector4<T> &aOther) const
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpgtu4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<=(const Vector4<T> &aOther) const
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpleu4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<(const Vector4<T> &aOther) const
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpltu4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator!=(const Vector4<T> &aOther) const
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpne4(*this, aOther) != 0U;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator==(const Vector4<T> &aOther) const
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpeq4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>=(const Vector4<T> &aOther) const
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpges4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>(const Vector4<T> &aOther) const
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpgts4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<=(const Vector4<T> &aOther) const
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmples4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<(const Vector4<T> &aOther) const
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmplts4(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator!=(const Vector4<T> &aOther) const
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpne4(*this, aOther) != 0U;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator==(const Vector4<T> &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);

    return (__vcmpeq2(leftPtr[0], rightPtr[0]) & __vcmpeq2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>=(const Vector4<T> &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpges2(leftPtr[0], rightPtr[0]) & __vcmpges2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>(const Vector4<T> &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpgts2(leftPtr[0], rightPtr[0]) & __vcmpgts2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<=(const Vector4<T> &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmples2(leftPtr[0], rightPtr[0]) & __vcmples2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<(const Vector4<T> &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmplts2(leftPtr[0], rightPtr[0]) & __vcmplts2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator!=(const Vector4<T> &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpne2(leftPtr[0], rightPtr[0]) | __vcmpne2(leftPtr[1], rightPtr[1])) != 0U;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator==(const Vector4<T> &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpeq2(leftPtr[0], rightPtr[0]) & __vcmpeq2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>=(const Vector4<T> &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpgeu2(leftPtr[0], rightPtr[0]) & __vcmpgeu2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>(const Vector4<T> &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpgtu2(leftPtr[0], rightPtr[0]) & __vcmpgtu2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<=(const Vector4<T> &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpleu2(leftPtr[0], rightPtr[0]) & __vcmpleu2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<(const Vector4<T> &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpltu2(leftPtr[0], rightPtr[0]) & __vcmpltu2(leftPtr[1], rightPtr[1])) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator!=(const Vector4<T> &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(this);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aOther);
    return (__vcmpne2(leftPtr[0], rightPtr[0]) | __vcmpne2(leftPtr[1], rightPtr[1])) != 0U;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator==(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    return __hbeq2(leftPtr[0], rightPtr[0]) && __hbeq2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>=(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    return __hbge2(leftPtr[0], rightPtr[0]) && __hbge2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    return __hbgt2(leftPtr[0], rightPtr[0]) && __hbgt2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<=(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    return __hble2(leftPtr[0], rightPtr[0]) && __hble2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    return __hblt2(leftPtr[0], rightPtr[0]) && __hblt2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator!=(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    // __hbne2 returns true only if both elements are != but we need true if any element is !=
    // so we use hbeq and negate the result
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    return !(__hbeq2(leftPtr[0], rightPtr[0]) && __hbeq2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator==(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aOther);
    return __hbeq2(leftPtr[0], rightPtr[0]) && __hbeq2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>=(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aOther);
    return __hbge2(leftPtr[0], rightPtr[0]) && __hbge2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator>(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aOther);
    return __hbgt2(leftPtr[0], rightPtr[0]) && __hbgt2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<=(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aOther);
    return __hble2(leftPtr[0], rightPtr[0]) && __hble2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator<(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aOther);
    return __hblt2(leftPtr[0], rightPtr[0]) && __hblt2(leftPtr[1], rightPtr[1]);
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector4<T>::operator!=(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    // __hbne2 returns true only if both elements are != but we need true if any element is !=
    // so we use hbeq and negate the result
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aOther);
    return !(__hbeq2(leftPtr[0], rightPtr[0]) && __hbeq2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Negation
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::operator-() const
    requires RealSignedNumber<T> || ComplexNumber<T>
{
    return Vector4<T>(-x, -y, -z, -w);
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-() const
    requires IsSByte<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vnegss4(*this));
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-() const
    requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *temp = reinterpret_cast<const uint *>(this);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vnegss2(temp[0]);
    resPtr[1]    = __vnegss2(temp[1]);
    return res;
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-() const
    requires IsBFloat16<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *temp = reinterpret_cast<const nv_bfloat162 *>(this);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __hneg2(temp[0]);
    resPtr[1]            = __hneg2(temp[1]);
    return res;
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-() const
    requires IsHalfFp16<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *temp = reinterpret_cast<const half2 *>(this);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __hneg2(temp[0]);
    resPtr[1]     = __hneg2(temp[1]);
    return res;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator+=(T aOther)
{
    x += aOther;
    y += aOther;
    z += aOther;
    w += aOther;
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::operator+=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x += aOther;
    y += aOther;
    z += aOther;
    w += aOther;
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &aOther)
{
    x += aOther.x;
    y += aOther.y;
    z += aOther.z;
    w += aOther.w;
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &aOther)
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vaddus4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &aOther)
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vaddss4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    uint *thisPtr        = reinterpret_cast<uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);

    thisPtr[0] = __vaddus2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __vaddus2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    uint *thisPtr        = reinterpret_cast<uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);

    thisPtr[0] = __vaddss2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __vaddss2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr        = reinterpret_cast<nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);

    thisPtr[0] = __hadd2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __hadd2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator+=(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);

    thisPtr[0] = __hadd2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __hadd2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> Vector4<T>::operator+(const Vector4<T> &aOther) const
{
    return Vector4<T>{T(x + aOther.x), T(y + aOther.y), T(z + aOther.z), T(w + aOther.w)};
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator+(const Vector4<T> &aOther) const
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vaddus4(*this, aOther));
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator+(const Vector4<T> &aOther) const
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vaddss4(*this, aOther));
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator+(const Vector4<T> &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *thisPtr  = reinterpret_cast<const uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vaddss2(thisPtr[0], otherPtr[0]);
    resPtr[1]    = __vaddss2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator+(const Vector4<T> &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *thisPtr  = reinterpret_cast<const uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vaddus2(thisPtr[0], otherPtr[0]);
    resPtr[1]    = __vaddus2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator+(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *thisPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __hadd2(thisPtr[0], otherPtr[0]);
    resPtr[1]            = __hadd2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator+(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *thisPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __hadd2(thisPtr[0], otherPtr[0]);
    resPtr[1]     = __hadd2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator-=(T aOther)
{
    x -= aOther;
    y -= aOther;
    z -= aOther;
    w -= aOther;
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::operator-=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x -= aOther;
    y -= aOther;
    z -= aOther;
    w -= aOther;
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &aOther)
{
    x -= aOther.x;
    y -= aOther.y;
    z -= aOther.z;
    w -= aOther.w;
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &aOther)
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubus4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &aOther)
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubss4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    uint *thisPtr        = reinterpret_cast<uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);

    thisPtr[0] = __vsubus2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __vsubus2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    uint *thisPtr        = reinterpret_cast<uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);

    thisPtr[0] = __vsubss2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __vsubss2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr        = reinterpret_cast<nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);

    thisPtr[0] = __hsub2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __hsub2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator-=(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);

    thisPtr[0] = __hsub2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __hsub2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::SubInv(const Vector4<T> &aOther)
{
    x = aOther.x - x;
    y = aOther.y - y;
    z = aOther.z - z;
    w = aOther.w - w;
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::SubInv(const Vector4<T> &aOther)
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubus4(aOther, *this));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::SubInv(const Vector4<T> &aOther)
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubss4(aOther, *this));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::SubInv(const Vector4<T> &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    uint *thisPtr        = reinterpret_cast<uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);

    thisPtr[0] = __vsubus2(otherPtr[0], thisPtr[0]);
    thisPtr[1] = __vsubus2(otherPtr[1], thisPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::SubInv(const Vector4<T> &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    uint *thisPtr        = reinterpret_cast<uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);

    thisPtr[0] = __vsubss2(otherPtr[0], thisPtr[0]);
    thisPtr[1] = __vsubss2(otherPtr[1], thisPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::SubInv(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr        = reinterpret_cast<nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);

    thisPtr[0] = __hsub2(otherPtr[0], thisPtr[0]);
    thisPtr[1] = __hsub2(otherPtr[1], thisPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::SubInv(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);

    thisPtr[0] = __hsub2(otherPtr[0], thisPtr[0]);
    thisPtr[1] = __hsub2(otherPtr[1], thisPtr[1]);
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> Vector4<T>::operator-(const Vector4<T> &aOther) const
{
    return Vector4<T>{T(x - aOther.x), T(y - aOther.y), T(z - aOther.z), T(w - aOther.w)};
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-(const Vector4<T> &aOther) const
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vsubus4(*this, aOther));
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-(const Vector4<T> &aOther) const
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vsubss4(*this, aOther));
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-(const Vector4<T> &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *thisPtr  = reinterpret_cast<const uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vsubss2(thisPtr[0], otherPtr[0]);
    resPtr[1]    = __vsubss2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-(const Vector4<T> &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *thisPtr  = reinterpret_cast<const uint *>(this);
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vsubus2(thisPtr[0], otherPtr[0]);
    resPtr[1]    = __vsubus2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *thisPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __hsub2(thisPtr[0], otherPtr[0]);
    resPtr[1]            = __hsub2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator-(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *thisPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __hsub2(thisPtr[0], otherPtr[0]);
    resPtr[1]     = __hsub2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator*=(T aOther)
{
    x *= aOther;
    y *= aOther;
    z *= aOther;
    w *= aOther;
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::operator*=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x *= aOther;
    y *= aOther;
    z *= aOther;
    w *= aOther;
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator*=(const Vector4<T> &aOther)
{
    x *= aOther.x;
    y *= aOther.y;
    z *= aOther.z;
    w *= aOther.w;
    return *this;
}

/// <summary>
/// Component wise multiplication SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator*=(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr        = reinterpret_cast<nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);

    thisPtr[0] = __hmul2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __hmul2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise multiplication SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator*=(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);

    thisPtr[0] = __hmul2(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __hmul2(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> Vector4<T>::operator*(const Vector4<T> &aOther) const
{
    return Vector4<T>{T(x * aOther.x), T(y * aOther.y), T(z * aOther.z), T(w * aOther.w)};
}

/// <summary>
/// Component wise multiplication SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator*(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *thisPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __hmul2(thisPtr[0], otherPtr[0]);
    resPtr[1]            = __hmul2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise multiplication SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator*(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *thisPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __hmul2(thisPtr[0], otherPtr[0]);
    resPtr[1]     = __hmul2(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator/=(T aOther)
{
    x /= aOther;
    y /= aOther;
    z /= aOther;
    w /= aOther;
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::operator/=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x /= aOther;
    y /= aOther;
    z /= aOther;
    w /= aOther;
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::operator/=(const Vector4<T> &aOther)
{
    x /= aOther.x;
    y /= aOther.y;
    z /= aOther.z;
    w /= aOther.w;
    return *this;
}

/// <summary>
/// Component wise division SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator/=(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr        = reinterpret_cast<nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);

    thisPtr[0] = __h2div(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __h2div(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise division SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::operator/=(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);

    thisPtr[0] = __h2div(thisPtr[0], otherPtr[0]);
    thisPtr[1] = __h2div(thisPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise division (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::DivInv(const Vector4<T> &aOther)
{
    x = aOther.x / x;
    y = aOther.y / y;
    z = aOther.z / z;
    w = aOther.w / w;
    return *this;
}

/// <summary>
/// Component wise division SIMD (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::DivInv(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr        = reinterpret_cast<nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);

    thisPtr[0] = __h2div(otherPtr[0], thisPtr[0]);
    thisPtr[1] = __h2div(otherPtr[1], thisPtr[1]);
    return *this;
}

/// <summary>
/// Component wise division SIMD (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::DivInv(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);

    thisPtr[0] = __h2div(otherPtr[0], thisPtr[0]);
    thisPtr[1] = __h2div(otherPtr[1], thisPtr[1]);
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> Vector4<T>::operator/(const Vector4<T> &aOther) const
{
    return Vector4<T>{T(x / aOther.x), T(y / aOther.y), T(z / aOther.z), T(w / aOther.w)};
}

/// <summary>
/// Component wise division SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator/(const Vector4<T> &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *thisPtr  = reinterpret_cast<const nv_bfloat162 *>(this);
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __h2div(thisPtr[0], otherPtr[0]);
    resPtr[1]            = __h2div(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Component wise division SIMD
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::operator/(const Vector4<T> &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *thisPtr  = reinterpret_cast<const half2 *>(this);
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __h2div(thisPtr[0], otherPtr[0]);
    resPtr[1]     = __h2div(thisPtr[1], otherPtr[1]);
    return res;
}

/// <summary>
/// Inplace integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivRound(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTiesAwayFromZero(x, aOther.x);
    y = DivRoundTiesAwayFromZero(y, aOther.y);
    z = DivRoundTiesAwayFromZero(z, aOther.z);
    w = DivRoundTiesAwayFromZero(w, aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivRoundNearest(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundNearestEven(x, aOther.x);
    y = DivRoundNearestEven(y, aOther.y);
    z = DivRoundNearestEven(z, aOther.z);
    w = DivRoundNearestEven(w, aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivRoundZero(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardZero(x, aOther.x);
    y = DivRoundTowardZero(y, aOther.y);
    z = DivRoundTowardZero(z, aOther.z);
    w = DivRoundTowardZero(w, aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivFloor(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardNegInf(x, aOther.x);
    y = DivRoundTowardNegInf(y, aOther.y);
    z = DivRoundTowardNegInf(z, aOther.z);
    w = DivRoundTowardNegInf(w, aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivCeil(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardPosInf(x, aOther.x);
    y = DivRoundTowardPosInf(y, aOther.y);
    z = DivRoundTowardPosInf(z, aOther.z);
    w = DivRoundTowardPosInf(w, aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvRound(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTiesAwayFromZero(aOther.x, x);
    y = DivRoundTiesAwayFromZero(aOther.y, y);
    z = DivRoundTiesAwayFromZero(aOther.z, z);
    w = DivRoundTiesAwayFromZero(aOther.w, w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (inverted inplace div: this =
/// aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvRoundNearest(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundNearestEven(aOther.x, x);
    y = DivRoundNearestEven(aOther.y, y);
    z = DivRoundNearestEven(aOther.z, z);
    w = DivRoundNearestEven(aOther.w, w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (inverted inplace div: this = aOther /
/// this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvRoundZero(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardZero(aOther.x, x);
    y = DivRoundTowardZero(aOther.y, y);
    z = DivRoundTowardZero(aOther.z, z);
    w = DivRoundTowardZero(aOther.w, w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvFloor(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardNegInf(aOther.x, x);
    y = DivRoundTowardNegInf(aOther.y, y);
    z = DivRoundTowardNegInf(aOther.z, z);
    w = DivRoundTowardNegInf(aOther.w, w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvCeil(const Vector4 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardPosInf(aOther.x, x);
    y = DivRoundTowardPosInf(aOther.y, y);
    z = DivRoundTowardPosInf(aOther.z, z);
    w = DivRoundTowardPosInf(aOther.w, w);
    return *this;
}

/// <summary>
/// Integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivRound(const Vector4 &aLeft, const Vector4 &aRight)
    requires RealIntegral<T>
{
    return Vector4<T>{DivRoundTiesAwayFromZero(aLeft.x, aRight.x), DivRoundTiesAwayFromZero(aLeft.y, aRight.y),
                      DivRoundTiesAwayFromZero(aLeft.z, aRight.z), DivRoundTiesAwayFromZero(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivRoundNearest(const Vector4 &aLeft, const Vector4 &aRight)
    requires RealIntegral<T>
{
    return Vector4<T>{DivRoundNearestEven(aLeft.x, aRight.x), DivRoundNearestEven(aLeft.y, aRight.y),
                      DivRoundNearestEven(aLeft.z, aRight.z), DivRoundNearestEven(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivRoundZero(const Vector4 &aLeft, const Vector4 &aRight)
    requires RealIntegral<T>
{
    return Vector4<T>{DivRoundTowardZero(aLeft.x, aRight.x), DivRoundTowardZero(aLeft.y, aRight.y),
                      DivRoundTowardZero(aLeft.z, aRight.z), DivRoundTowardZero(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivFloor(const Vector4 &aLeft, const Vector4 &aRight)
    requires RealIntegral<T>
{
    return Vector4<T>{DivRoundTowardNegInf(aLeft.x, aRight.x), DivRoundTowardNegInf(aLeft.y, aRight.y),
                      DivRoundTowardNegInf(aLeft.z, aRight.z), DivRoundTowardNegInf(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivCeil(const Vector4 &aLeft, const Vector4 &aRight)
    requires RealIntegral<T>
{
    return Vector4<T>{DivRoundTowardPosInf(aLeft.x, aRight.x), DivRoundTowardPosInf(aLeft.y, aRight.y),
                      DivRoundTowardPosInf(aLeft.z, aRight.z), DivRoundTowardPosInf(aLeft.w, aRight.w)};
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivScaleRoundNearest(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundNearestEven(x, aScale);
    y = DivScaleRoundNearestEven(y, aScale);
    z = DivScaleRoundNearestEven(z, aScale);
    w = DivScaleRoundNearestEven(w, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivRound(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRound(aOther.x);
    y.DivRound(aOther.y);
    z.DivRound(aOther.z);
    w.DivRound(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivRoundNearest(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRoundNearest(aOther.x);
    y.DivRoundNearest(aOther.y);
    z.DivRoundNearest(aOther.z);
    w.DivRoundNearest(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivRoundZero(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRoundZero(aOther.x);
    y.DivRoundZero(aOther.y);
    z.DivRoundZero(aOther.z);
    w.DivRoundZero(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivFloor(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivFloor(aOther.x);
    y.DivFloor(aOther.y);
    z.DivFloor(aOther.z);
    w.DivFloor(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivCeil(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivCeil(aOther.x);
    y.DivCeil(aOther.y);
    z.DivCeil(aOther.z);
    w.DivCeil(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvRound(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRound(aOther.x);
    y.DivInvRound(aOther.y);
    z.DivInvRound(aOther.z);
    w.DivInvRound(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (inverted inplace div: this =
/// aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvRoundNearest(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRoundNearest(aOther.x);
    y.DivInvRoundNearest(aOther.y);
    z.DivInvRoundNearest(aOther.z);
    w.DivInvRoundNearest(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (inverted inplace div: this = aOther /
/// this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvRoundZero(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRoundZero(aOther.x);
    y.DivInvRoundZero(aOther.y);
    z.DivInvRoundZero(aOther.z);
    w.DivInvRoundZero(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvFloor(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvFloor(aOther.x);
    y.DivInvFloor(aOther.y);
    z.DivInvFloor(aOther.z);
    w.DivInvFloor(aOther.w);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivInvCeil(const Vector4 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvCeil(aOther.x);
    y.DivInvCeil(aOther.y);
    z.DivInvCeil(aOther.z);
    w.DivInvCeil(aOther.w);
    return *this;
}

/// <summary>
/// Integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivRound(const Vector4 &aLeft, const Vector4 &aRight)
    requires ComplexIntegral<T>
{
    return Vector4<T>{T::DivRound(aLeft.x, aRight.x), T::DivRound(aLeft.y, aRight.y), T::DivRound(aLeft.z, aRight.z),
                      T::DivRound(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivRoundNearest(const Vector4 &aLeft, const Vector4 &aRight)
    requires ComplexIntegral<T>
{
    return Vector4<T>{T::DivRoundNearest(aLeft.x, aRight.x), T::DivRoundNearest(aLeft.y, aRight.y),
                      T::DivRoundNearest(aLeft.z, aRight.z), T::DivRoundNearest(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivRoundZero(const Vector4 &aLeft, const Vector4 &aRight)
    requires ComplexIntegral<T>
{
    return Vector4<T>{T::DivRoundZero(aLeft.x, aRight.x), T::DivRoundZero(aLeft.y, aRight.y),
                      T::DivRoundZero(aLeft.z, aRight.z), T::DivRoundZero(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivFloor(const Vector4 &aLeft, const Vector4 &aRight)
    requires ComplexIntegral<T>
{
    return Vector4<T>{T::DivFloor(aLeft.x, aRight.x), T::DivFloor(aLeft.y, aRight.y), T::DivFloor(aLeft.z, aRight.z),
                      T::DivFloor(aLeft.w, aRight.w)};
}

/// <summary>
/// Integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::DivCeil(const Vector4 &aLeft, const Vector4 &aRight)
    requires ComplexIntegral<T>
{
    return Vector4<T>{T::DivCeil(aLeft.x, aRight.x), T::DivCeil(aLeft.y, aRight.y), T::DivCeil(aLeft.z, aRight.z),
                      T::DivCeil(aLeft.w, aRight.w)};
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::DivScaleRoundNearest(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleRoundNearest(aScale);
    y.DivScaleRoundNearest(aScale);
    z.DivScaleRoundNearest(aScale);
    w.DivScaleRoundNearest(aScale);
    return *this;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
DEVICE_CODE const T &Vector4<T>::operator[](Axis4D aAxis) const
    requires DeviceCode<T>
{
    switch (aAxis)
    {
        case Axis4D::X:
            return x;
        case Axis4D::Y:
            return y;
        case Axis4D::Z:
            return z;
        case Axis4D::W:
            return w;
    }
    return x;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
const T &Vector4<T>::operator[](Axis4D aAxis) const
    requires HostCode<T>
{
    switch (aAxis)
    {
        case Axis4D::X:
            return x;
        case Axis4D::Y:
            return y;
        case Axis4D::Z:
            return z;
        case Axis4D::W:
            return w;
    }

    throw INVALIDARGUMENT(aAxis, aAxis);
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
DEVICE_CODE T &Vector4<T>::operator[](Axis4D aAxis)
    requires DeviceCode<T>
{
    switch (aAxis)
    {
        case Axis4D::X:
            return x;
        case Axis4D::Y:
            return y;
        case Axis4D::Z:
            return z;
        case Axis4D::W:
            return w;
    }
    return x;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
T &Vector4<T>::operator[](Axis4D aAxis)
    requires HostCode<T>
{
    switch (aAxis)
    {
        case Axis4D::X:
            return x;
        case Axis4D::Y:
            return y;
        case Axis4D::Z:
            return z;
        case Axis4D::W:
            return w;
    }

    throw INVALIDARGUMENT(aAxis, aAxis);
}
#pragma endregion

#pragma region Convert Methods
/// <summary>
/// Type conversion without saturation, direct type conversion
/// </summary>
template <Number T> template <Number T2> Vector4<T> DEVICE_CODE Vector4<T>::Convert(const Vector4<T2> &aVec)
{
    return {StaticCast<T2, T>(aVec.x), StaticCast<T2, T>(aVec.y), StaticCast<T2, T>(aVec.z), StaticCast<T2, T>(aVec.w)};
}
#pragma endregion

#pragma region Integral only methods
/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::LShift(const Vector4<T> &aOther)
    requires RealIntegral<T>
{
    x = x << aOther.x; // NOLINT
    y = y << aOther.y; // NOLINT
    z = z << aOther.z; // NOLINT
    w = w << aOther.w; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::LShift(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealIntegral<T>
{
    Vector4<T> ret;              // NOLINT
    ret.x = aLeft.x << aRight.x; // NOLINT
    ret.y = aLeft.y << aRight.y; // NOLINT
    ret.z = aLeft.z << aRight.z; // NOLINT
    ret.w = aLeft.w << aRight.w; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::RShift(const Vector4<T> &aOther)
    requires RealIntegral<T>
{
    x = x >> aOther.x; // NOLINT
    y = y >> aOther.y; // NOLINT
    z = z >> aOther.z; // NOLINT
    w = w >> aOther.w; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::RShift(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealIntegral<T>
{
    Vector4<T> ret;              // NOLINT
    ret.x = aLeft.x >> aRight.x; // NOLINT
    ret.y = aLeft.y >> aRight.y; // NOLINT
    ret.z = aLeft.z >> aRight.z; // NOLINT
    ret.w = aLeft.w >> aRight.w; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::LShift(uint aOther)
    requires RealIntegral<T>
{
    x = x << aOther; // NOLINT
    y = y << aOther; // NOLINT
    z = z << aOther; // NOLINT
    w = w << aOther; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::LShift(const Vector4<T> &aLeft, uint aRight)
    requires RealIntegral<T>
{
    Vector4<T> ret;            // NOLINT
    ret.x = aLeft.x << aRight; // NOLINT
    ret.y = aLeft.y << aRight; // NOLINT
    ret.z = aLeft.z << aRight; // NOLINT
    ret.w = aLeft.w << aRight; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::RShift(uint aOther)
    requires RealIntegral<T>
{
    x = x >> aOther; // NOLINT
    y = y >> aOther; // NOLINT
    z = z >> aOther; // NOLINT
    w = w >> aOther; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::RShift(const Vector4<T> &aLeft, uint aRight)
    requires RealIntegral<T>
{
    Vector4<T> ret;            // NOLINT
    ret.x = aLeft.x >> aRight; // NOLINT
    ret.y = aLeft.y >> aRight; // NOLINT
    ret.z = aLeft.z >> aRight; // NOLINT
    ret.w = aLeft.w >> aRight; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise And
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::And(const Vector4<T> &aOther)
    requires RealIntegral<T>
{
    x = x & aOther.x; // NOLINT
    y = y & aOther.y; // NOLINT
    z = z & aOther.z; // NOLINT
    w = w & aOther.w; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise And
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::And(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealIntegral<T>
{
    Vector4<T> ret;             // NOLINT
    ret.x = aLeft.x & aRight.x; // NOLINT
    ret.y = aLeft.y & aRight.y; // NOLINT
    ret.z = aLeft.z & aRight.z; // NOLINT
    ret.w = aLeft.w & aRight.w; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise Or
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Or(const Vector4<T> &aOther)
    requires RealIntegral<T>
{
    x = x | aOther.x; // NOLINT
    y = y | aOther.y; // NOLINT
    z = z | aOther.z; // NOLINT
    w = w | aOther.w; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise Or
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Or(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealIntegral<T>
{
    Vector4<T> ret;             // NOLINT
    ret.x = aLeft.x | aRight.x; // NOLINT
    ret.y = aLeft.y | aRight.y; // NOLINT
    ret.z = aLeft.z | aRight.z; // NOLINT
    ret.w = aLeft.w | aRight.w; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise Xor
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Xor(const Vector4<T> &aOther)
    requires RealIntegral<T>
{
    x = x ^ aOther.x; // NOLINT
    y = y ^ aOther.y; // NOLINT
    z = z ^ aOther.z; // NOLINT
    w = w ^ aOther.w; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise Xor
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Xor(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealIntegral<T>
{
    Vector4<T> ret;             // NOLINT
    ret.x = aLeft.x ^ aRight.x; // NOLINT
    ret.y = aLeft.y ^ aRight.y; // NOLINT
    ret.z = aLeft.z ^ aRight.z; // NOLINT
    ret.w = aLeft.w ^ aRight.w; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise negation
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Not()
    requires RealIntegral<T>
{
    x = ~x; // NOLINT
    y = ~y; // NOLINT
    z = ~z; // NOLINT
    w = ~w; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise negation
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Not(const Vector4<T> &aVec)
    requires RealIntegral<T>
{
    Vector4<T> ret;  // NOLINT
    ret.x = ~aVec.x; // NOLINT
    ret.y = ~aVec.y; // NOLINT
    ret.z = ~aVec.z; // NOLINT
    ret.w = ~aVec.w; // NOLINT
    return ret;
}
#pragma endregion

#pragma region Methods
#pragma region Exp
/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Exp()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::exp(x);
    y = std::exp(y);
    z = std::exp(z);
    w = std::exp(w);
    return *this;
}
/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Exp()
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>
{
    x = T::Exp(x);
    y = T::Exp(y);
    z = T::Exp(z);
    w = T::Exp(w);
    return *this;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Exp()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = exp(x);
    y = exp(y);
    z = exp(z);
    w = exp(w);
    return *this;
}

/// <summary>
/// Element wise exponential (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Exp()
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = h2exp(thisPtr[0]);
    thisPtr[1]            = h2exp(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise exponential (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Exp()
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr = reinterpret_cast<half2 *>(this);
    thisPtr[0]     = h2exp(thisPtr[0]);
    thisPtr[1]     = h2exp(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Exp(const Vector4<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = exp(aVec.x);
    ret.y = exp(aVec.y);
    ret.z = exp(aVec.z);
    ret.w = exp(aVec.w);
    return ret;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Exp(const Vector4<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = std::exp(aVec.x);
    ret.y = std::exp(aVec.y);
    ret.z = std::exp(aVec.z);
    ret.w = std::exp(aVec.w);
    return ret;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Exp(const Vector4<T> &aVec)
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = T::Exp(aVec.x);
    ret.y = T::Exp(aVec.y);
    ret.z = T::Exp(aVec.z);
    ret.w = T::Exp(aVec.w);
    return ret;
}

/// <summary>
/// Element wise exponential (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Exp(const Vector4<T> &aVec)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    Vector4<T> ret; // NOLINT
    const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
    nv_bfloat162 *retPtr        = reinterpret_cast<nv_bfloat162 *>(&ret);
    retPtr[0]                   = h2exp(aVecPtr[0]);
    retPtr[1]                   = h2exp(aVecPtr[1]);
    return ret;
}

/// <summary>
/// Element wise exponential (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Exp(const Vector4<T> &aVec)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    Vector4<T> ret; // NOLINT
    const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
    half2 *retPtr        = reinterpret_cast<half2 *>(&ret);
    retPtr[0]            = h2exp(aVecPtr[0]);
    retPtr[1]            = h2exp(aVecPtr[1]);
    return ret;
}

#pragma endregion

#pragma region Log
/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Ln()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::log(x);
    y = std::log(y);
    z = std::log(z);
    w = std::log(w);
    return *this;
}
/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Ln()
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    x = T::Ln(x);
    y = T::Ln(y);
    z = T::Ln(z);
    w = T::Ln(w);
    return *this;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Ln()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = log(x);
    y = log(y);
    z = log(z);
    w = log(w);
    return *this;
}

/// <summary>
/// Element wise natural logarithm (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Ln()
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = h2log(thisPtr[0]);
    thisPtr[1]            = h2log(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise natural logarithm (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Ln()
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr = reinterpret_cast<half2 *>(this);
    thisPtr[0]     = h2log(thisPtr[0]);
    thisPtr[1]     = h2log(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Ln(const Vector4<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = log(aVec.x);
    ret.y = log(aVec.y);
    ret.z = log(aVec.z);
    ret.w = log(aVec.w);
    return ret;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Ln(const Vector4<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = std::log(aVec.x);
    ret.y = std::log(aVec.y);
    ret.z = std::log(aVec.z);
    ret.w = std::log(aVec.w);
    return ret;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Ln(const Vector4<T> &aVec)
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = T::Ln(aVec.x);
    ret.y = T::Ln(aVec.y);
    ret.z = T::Ln(aVec.z);
    ret.w = T::Ln(aVec.w);
    return ret;
}

/// <summary>
/// Element wise natural logarithm (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Ln(const Vector4<T> &aVec)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    Vector4<T> ret; // NOLINT
    const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
    nv_bfloat162 *retPtr        = reinterpret_cast<nv_bfloat162 *>(&ret);
    retPtr[0]                   = h2log(aVecPtr[0]);
    retPtr[1]                   = h2log(aVecPtr[1]);
    return ret;
}

/// <summary>
/// Element wise natural logarithm (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Ln(const Vector4<T> &aVec)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    Vector4<T> ret; // NOLINT
    const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
    half2 *retPtr        = reinterpret_cast<half2 *>(&ret);
    retPtr[0]            = h2log(aVecPtr[0]);
    retPtr[1]            = h2log(aVecPtr[1]);
    return ret;
}

#pragma endregion

#pragma region Sqr
/// <summary>
/// Element wise square
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> &Vector4<T>::Sqr()
{
    x = x * x;
    y = y * y;
    z = z * z;
    w = w * w;
    return *this;
}

/// <summary>
/// Element wise square
/// </summary>
template <Number T> DEVICE_CODE Vector4<T> Vector4<T>::Sqr(const Vector4<T> &aVec)
{
    Vector4<T> ret; // NOLINT
    ret.x = aVec.x * aVec.x;
    ret.y = aVec.y * aVec.y;
    ret.z = aVec.z * aVec.z;
    ret.w = aVec.w * aVec.w;
    return ret;
}

#pragma endregion

#pragma region Sqrt
/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Sqrt()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::sqrt(x);
    y = std::sqrt(y);
    z = std::sqrt(z);
    w = std::sqrt(w);
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Sqrt()
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    x = T::Sqrt(x);
    y = T::Sqrt(y);
    z = T::Sqrt(z);
    w = T::Sqrt(w);
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Sqrt()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = sqrt(x);
    y = sqrt(y);
    z = sqrt(z);
    w = sqrt(w);
    return *this;
}

/// <summary>
/// Element wise square root (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Sqrt()
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = h2sqrt(thisPtr[0]);
    thisPtr[1]            = h2sqrt(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise square root (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Sqrt()
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *thisPtr = reinterpret_cast<half2 *>(this);
    thisPtr[0]     = h2sqrt(thisPtr[0]);
    thisPtr[1]     = h2sqrt(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Sqrt(const Vector4<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = sqrt(aVec.x);
    ret.y = sqrt(aVec.y);
    ret.z = sqrt(aVec.z);
    ret.w = sqrt(aVec.w);
    return ret;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Sqrt(const Vector4<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = std::sqrt(aVec.x);
    ret.y = std::sqrt(aVec.y);
    ret.z = std::sqrt(aVec.z);
    ret.w = std::sqrt(aVec.w);
    return ret;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Sqrt(const Vector4<T> &aVec)
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = T::Sqrt(aVec.x);
    ret.y = T::Sqrt(aVec.y);
    ret.z = T::Sqrt(aVec.z);
    ret.w = T::Sqrt(aVec.w);
    return ret;
}

/// <summary>
/// Element wise square root (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Sqrt(const Vector4<T> &aVec)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    Vector4<T> ret; // NOLINT
    const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
    nv_bfloat162 *retPtr        = reinterpret_cast<nv_bfloat162 *>(&ret);
    retPtr[0]                   = h2sqrt(aVecPtr[0]);
    retPtr[1]                   = h2sqrt(aVecPtr[1]);
    return ret;
}

/// <summary>
/// Element wise square root (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Sqrt(const Vector4<T> &aVec)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    Vector4<T> ret; // NOLINT
    const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
    half2 *retPtr        = reinterpret_cast<half2 *>(&ret);
    retPtr[0]            = h2sqrt(aVecPtr[0]);
    retPtr[1]            = h2sqrt(aVecPtr[1]);
    return ret;
}
#pragma endregion

#pragma region Abs
/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Abs()
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    if constexpr (sizeof(T) < 4)
    {
        // short and sbyte are computed as int
        x = static_cast<T>(std::abs(x));
        y = static_cast<T>(std::abs(y));
        z = static_cast<T>(std::abs(z));
        w = static_cast<T>(std::abs(w));
    }
    else
    {
        x = std::abs(x);
        y = std::abs(y);
        z = std::abs(z);
        w = std::abs(w);
    }
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Abs()
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    x = T::Abs(x);
    y = T::Abs(y);
    z = T::Abs(z);
    w = T::Abs(w);
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Abs()
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = abs(x);
    y = abs(y);
    z = abs(z);
    w = abs(w);
    return *this;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Abs()
    requires IsSByte<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vabsss4(*this));
    return *this;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Abs()
    requires IsShort<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>
{
    uint *temp = reinterpret_cast<uint *>(this);
    temp[0]    = __vabsss2(temp[0]);
    temp[1]    = __vabsss2(temp[1]);
    return *this;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Abs()
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    nv_bfloat162 *temp = reinterpret_cast<nv_bfloat162 *>(this);
    temp[0]            = __habs2(temp[0]);
    temp[1]            = __habs2(temp[1]);
    return *this;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Abs()
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    half2 *temp = reinterpret_cast<half2 *>(this);
    temp[0]     = __habs2(temp[0]);
    temp[1]     = __habs2(temp[1]);
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Abs(const Vector4<T> &aVec)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = abs(aVec.x);
    ret.y = abs(aVec.y);
    ret.z = abs(aVec.z);
    ret.w = abs(aVec.w);
    return ret;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Abs(const Vector4<T> &aVec)
    requires IsSByte<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>
{
    return FromUint(__vabsss4(aVec));
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Abs(const Vector4<T> &aVec)
    requires IsShort<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>
{
    const uint *temp = reinterpret_cast<const uint *>(&aVec);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vabsss2(temp[0]);
    resPtr[1]    = __vabsss2(temp[1]);
    return res;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Abs(const Vector4<T> &aVec)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *temp = reinterpret_cast<const nv_bfloat162 *>(&aVec);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __habs2(temp[0]);
    resPtr[1]            = __habs2(temp[1]);
    return res;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Abs(const Vector4<T> &aVec)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *temp = reinterpret_cast<const half2 *>(&aVec);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __habs2(temp[0]);
    resPtr[1]     = __habs2(temp[1]);
    return res;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Abs(const Vector4<T> &aVec)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    if constexpr (sizeof(T) < 4)
    {
        // short and sbyte are computed as int
        ret.x = static_cast<T>(std::abs(aVec.x));
        ret.y = static_cast<T>(std::abs(aVec.y));
        ret.z = static_cast<T>(std::abs(aVec.z));
        ret.w = static_cast<T>(std::abs(aVec.w));
    }
    else
    {
        ret.x = std::abs(aVec.x);
        ret.y = std::abs(aVec.y);
        ret.z = std::abs(aVec.z);
        ret.w = std::abs(aVec.w);
    }
    return ret;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Abs(const Vector4<T> &aVec)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = T::Abs(aVec.x);
    ret.y = T::Abs(aVec.y);
    ret.z = T::Abs(aVec.z);
    ret.w = T::Abs(aVec.w);
    return ret;
}
#pragma endregion

#pragma region AbsDiff
/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::AbsDiff(const Vector4<T> &aOther)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = std::abs(x - aOther.x);
    y = std::abs(y - aOther.y);
    z = std::abs(z - aOther.z);
    w = std::abs(w - aOther.w);
    return *this;
}

/// <summary>
/// Element wise absolute difference (per element also for complex!)
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::AbsDiff(const Vector4<T> &aOther)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    x = T::Abs(x - aOther.x);
    y = T::Abs(y - aOther.y);
    z = T::Abs(z - aOther.z);
    w = T::Abs(w - aOther.w);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = std::abs(aLeft.x - aRight.x);
    ret.y = std::abs(aLeft.y - aRight.y);
    ret.z = std::abs(aLeft.z - aRight.z);
    ret.w = std::abs(aLeft.w - aRight.w);
    return ret;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = T::Abs(aLeft.x - aRight.x);
    ret.y = T::Abs(aLeft.y - aRight.y);
    ret.z = T::Abs(aLeft.z - aRight.z);
    ret.w = T::Abs(aLeft.w - aRight.w);
    return ret;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::AbsDiff(const Vector4<T> &aOther)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = abs(x - aOther.x);
    y = abs(y - aOther.y);
    z = abs(z - aOther.z);
    w = abs(w - aOther.w);
    return *this;
}

// The SIMD versions do not saturate to max value!
/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::AbsDiff(const Vector4<T> &aOther)
    requires IsSByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vabsdiffs4(*this, aOther));
    return *this;
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::AbsDiff(const Vector4<T> &aOther)
    requires IsByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vabsdiffu4(*this, aOther));
    return *this;
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::AbsDiff(const Vector4<T> &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    uint *resPtr         = reinterpret_cast<uint *>(this);
    resPtr[0]            = __vabsdiffs2(resPtr[0], otherPtr[0]);
    resPtr[1]            = __vabsdiffs2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::AbsDiff(const Vector4<T> &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    uint *resPtr         = reinterpret_cast<uint *>(this);
    resPtr[0]            = __vabsdiffu2(resPtr[0], otherPtr[0]);
    resPtr[1]            = __vabsdiffu2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector4<T> ret; // NOLINT
    ret.x = abs(aLeft.x - aRight.x);
    ret.y = abs(aLeft.y - aRight.y);
    ret.z = abs(aLeft.z - aRight.z);
    ret.w = abs(aLeft.w - aRight.w);
    return ret;
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    return FromUint(__vabsdiffs4(aLeft, aRight));
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    return FromUint(__vabsdiffu4(aLeft, aRight));
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vabsdiffs2(leftPtr[0], rightPtr[0]);
    resPtr[1]    = __vabsdiffs2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vabsdiffu2(leftPtr[0], rightPtr[0]);
    resPtr[1]    = __vabsdiffu2(leftPtr[1], rightPtr[1]);
    return res;
}
#pragma endregion

#pragma region Methods for Complex types
/// <summary>
/// Conjugate complex per element
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Conj()
    requires ComplexNumber<T>
{
    x.Conj();
    y.Conj();
    z.Conj();
    w.Conj();
    return *this;
}

/// <summary>
/// Conjugate complex per element
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Conj(const Vector4<T> &aValue)
    requires ComplexNumber<T>
{
    return {T::Conj(aValue.x), T::Conj(aValue.y), T::Conj(aValue.z), T::Conj(aValue.w)};
}

/// <summary>
/// Conjugate complex multiplication: this * conj(aOther)  per element
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::ConjMul(const Vector4<T> &aOther)
    requires ComplexNumber<T>
{
    x.ConjMul(aOther.x);
    y.ConjMul(aOther.y);
    z.ConjMul(aOther.z);
    w.ConjMul(aOther.w);
    return *this;
}

/// <summary>
/// Conjugate complex multiplication: aLeft * conj(aRight) per element
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::ConjMul(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires ComplexNumber<T>
{
    return {T::ConjMul(aLeft.x, aRight.x), T::ConjMul(aLeft.y, aRight.y), T::ConjMul(aLeft.z, aRight.z),
            T::ConjMul(aLeft.w, aRight.w)};
}

/// <summary>
/// Complex magnitude per element
/// </summary>
template <Number T>
DEVICE_CODE Vector4<complex_basetype_t<T>> Vector4<T>::Magnitude() const
    requires ComplexFloatingPoint<T>
{
    Vector4<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.Magnitude();
    ret.y = y.Magnitude();
    ret.z = z.Magnitude();
    ret.w = w.Magnitude();
    return ret;
}

/// <summary>
/// Complex magnitude squared per element
/// </summary>
template <Number T>
DEVICE_CODE Vector4<complex_basetype_t<T>> Vector4<T>::MagnitudeSqr() const
    requires ComplexFloatingPoint<T>
{
    Vector4<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.MagnitudeSqr();
    ret.y = y.MagnitudeSqr();
    ret.z = z.MagnitudeSqr();
    ret.w = w.MagnitudeSqr();
    return ret;
}

/// <summary>
/// Angle between real and imaginary of a complex number (atan2(image, real)) per element
/// </summary>
template <Number T>
DEVICE_CODE Vector4<complex_basetype_t<T>> Vector4<T>::Angle() const
    requires ComplexFloatingPoint<T>
{
    Vector4<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.Angle();
    ret.y = y.Angle();
    ret.z = z.Angle();
    ret.w = w.Angle();
    return ret;
}
#pragma endregion

#pragma region Clamp
/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Clamp(T aMinVal, T aMaxVal)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = max(aMinVal, min(x, aMaxVal));
    y = max(aMinVal, min(y, aMaxVal));
    z = max(aMinVal, min(z, aMaxVal));
    w = max(aMinVal, min(w, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Clamp(T aMinVal, T aMaxVal)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::max(aMinVal, std::min(x, aMaxVal));
    y = std::max(aMinVal, std::min(y, aMaxVal));
    z = std::max(aMinVal, std::min(z, aMaxVal));
    w = std::max(aMinVal, std::min(w, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Clamp(T aMinVal, T aMaxVal)
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    x = T::Max(aMinVal, T::Min(x, aMaxVal));
    y = T::Max(aMinVal, T::Min(y, aMaxVal));
    z = T::Max(aMinVal, T::Min(z, aMaxVal));
    w = T::Max(aMinVal, T::Min(w, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
    requires ComplexNumber<T>
{
    x.Clamp(aMinVal, aMaxVal);
    y.Clamp(aMinVal, aMaxVal);
    z.Clamp(aMinVal, aMaxVal);
    w.Clamp(aMinVal, aMaxVal);
    return *this;
}

/// <summary>
/// Component wise clamp to maximum value range of given target type
/// </summary>
template <Number T>
template <Number TTarget>
DEVICE_CODE Vector4<T> &Vector4<T>::ClampToTargetType() noexcept
    requires(need_saturation_clamp_v<T, TTarget>)
{
    return Clamp(numeric_limits_conversion<T, TTarget>::lowest(), numeric_limits_conversion<T, TTarget>::max());
}

/// <summary>
/// Component wise clamp to maximum value range of given target type<para/>
/// NOP in case no saturation clamping is needed.
/// </summary>
template <Number T>
template <Number TTarget>
DEVICE_CODE Vector4<T> &Vector4<T>::ClampToTargetType() noexcept
    requires(!need_saturation_clamp_v<T, TTarget>)
{
    return *this;
}
#pragma endregion

#pragma region Min
/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = min(x, aRight.x);
    y = min(y, aRight.y);
    z = min(z, aRight.z);
    w = min(w, aRight.w);
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aOther)
    requires IsSByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vmins4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aOther)
    requires IsByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vminu4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    uint *resPtr         = reinterpret_cast<uint *>(this);
    resPtr[0]            = __vmins2(resPtr[0], otherPtr[0]);
    resPtr[1]            = __vmins2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    uint *resPtr         = reinterpret_cast<uint *>(this);
    resPtr[0]            = __vminu2(resPtr[0], otherPtr[0]);
    resPtr[1]            = __vminu2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    nv_bfloat162 *resPtr         = reinterpret_cast<nv_bfloat162 *>(this);
    resPtr[0]                    = __hmin2(resPtr[0], otherPtr[0]);
    resPtr[1]                    = __hmin2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);
    half2 *resPtr         = reinterpret_cast<half2 *>(this);
    resPtr[0]             = __hmin2(resPtr[0], otherPtr[0]);
    resPtr[1]             = __hmin2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Min(const Vector4<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::min(x, aRight.x);
    y = std::min(y, aRight.y);
    z = std::min(z, aRight.z);
    w = std::min(w, aRight.w);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Min(const Vector4<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    x.Min(aRight.x);
    y.Min(aRight.y);
    z.Min(aRight.z);
    w.Min(aRight.w);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    return Vector4<T>{T(min(aLeft.x, aRight.x)), T(min(aLeft.y, aRight.y)), T(min(aLeft.z, aRight.z)),
                      T(min(aLeft.w, aRight.w))};
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vmins4(aLeft, aRight));
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vminu4(aLeft, aRight));
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vmins2(leftPtr[0], rightPtr[0]);
    resPtr[1]    = __vmins2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vminu2(leftPtr[0], rightPtr[0]);
    resPtr[1]    = __vminu2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __hmin2(leftPtr[0], rightPtr[0]);
    resPtr[1]            = __hmin2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __hmin2(leftPtr[0], rightPtr[0]);
    resPtr[1]     = __hmin2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    return Vector4<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y), std::min(aLeft.z, aRight.z),
                      std::min(aLeft.w, aRight.w)};
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    return Vector4<T>{T::Min(aLeft.x, aRight.x), T::Min(aLeft.y, aRight.y), T::Min(aLeft.z, aRight.z),
                      T::Min(aLeft.w, aRight.w)};
}

/// <summary>
/// Returns the minimum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector4<T>::Min() const
    requires DeviceCode<T> && NativeNumber<T>
{
    return min(min(x, y), min(z, w));
}

/// <summary>
/// Returns the minimum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector4<T>::Min() const
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    return T::Min(T::Min(x, y), T::Min(z, w));
}

/// <summary>
/// Returns the minimum component of the vector
/// </summary>
template <Number T>
T Vector4<T>::Min() const
    requires HostCode<T> && NativeNumber<T>
{
    return std::min({x, y, z, w});
}
#pragma endregion

#pragma region Max
/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = max(x, aRight.x);
    y = max(y, aRight.y);
    z = max(z, aRight.z);
    w = max(w, aRight.w);
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aOther)
    requires IsSByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vmaxs4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aOther)
    requires IsByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vmaxu4(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aOther)
    requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    uint *resPtr         = reinterpret_cast<uint *>(this);
    resPtr[0]            = __vmaxs2(resPtr[0], otherPtr[0]);
    resPtr[1]            = __vmaxs2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aOther)
    requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *otherPtr = reinterpret_cast<const uint *>(&aOther);
    uint *resPtr         = reinterpret_cast<uint *>(this);
    resPtr[0]            = __vmaxu2(resPtr[0], otherPtr[0]);
    resPtr[1]            = __vmaxu2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aOther)
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *otherPtr = reinterpret_cast<const nv_bfloat162 *>(&aOther);
    nv_bfloat162 *resPtr         = reinterpret_cast<nv_bfloat162 *>(this);
    resPtr[0]                    = __hmax2(resPtr[0], otherPtr[0]);
    resPtr[1]                    = __hmax2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aOther)
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *otherPtr = reinterpret_cast<const half2 *>(&aOther);
    half2 *resPtr         = reinterpret_cast<half2 *>(this);
    resPtr[0]             = __hmax2(resPtr[0], otherPtr[0]);
    resPtr[1]             = __hmax2(resPtr[1], otherPtr[1]);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Max(const Vector4<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::max(x, aRight.x);
    y = std::max(y, aRight.y);
    z = std::max(z, aRight.z);
    w = std::max(w, aRight.w);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Max(const Vector4<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    x.Max(aRight.x);
    y.Max(aRight.y);
    z.Max(aRight.z);
    w.Max(aRight.w);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    return Vector4<T>{T(max(aLeft.x, aRight.x)), T(max(aLeft.y, aRight.y)), T(max(aLeft.z, aRight.z)),
                      T(max(aLeft.w, aRight.w))};
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vmaxs4(aLeft, aRight));
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vmaxu4(aLeft, aRight));
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vmaxs2(leftPtr[0], rightPtr[0]);
    resPtr[1]    = __vmaxs2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    Vector4<T> res;
    uint *resPtr = reinterpret_cast<uint *>(&res);
    resPtr[0]    = __vmaxu2(leftPtr[0], rightPtr[0]);
    resPtr[1]    = __vmaxu2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    Vector4<T> res;
    nv_bfloat162 *resPtr = reinterpret_cast<nv_bfloat162 *>(&res);
    resPtr[0]            = __hmax2(leftPtr[0], rightPtr[0]);
    resPtr[1]            = __hmax2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    Vector4<T> res;
    half2 *resPtr = reinterpret_cast<half2 *>(&res);
    resPtr[0]     = __hmax2(leftPtr[0], rightPtr[0]);
    resPtr[1]     = __hmax2(leftPtr[1], rightPtr[1]);
    return res;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    return Vector4<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y), std::max(aLeft.z, aRight.z),
                      std::max(aLeft.w, aRight.w)};
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    return Vector4<T>{T::Max(aLeft.x, aRight.x), T::Max(aLeft.y, aRight.y), T::Max(aLeft.z, aRight.z),
                      T::Max(aLeft.w, aRight.w)};
}

/// <summary>
/// Returns the maximum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector4<T>::Max() const
    requires DeviceCode<T> && NativeNumber<T>
{
    return max(max(x, y), max(z, w));
}

/// <summary>
/// Returns the maximum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector4<T>::Max() const
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    return T::Max(T::Max(x, y), T::Max(z, w));
}

/// <summary>
/// Returns the maximum component of the vector
/// </summary>
template <Number T>
T Vector4<T>::Max() const
    requires HostCode<T> && NativeNumber<T>
{
    return std::max({x, y, z, w});
}
#pragma endregion

#pragma region Round
/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Round(const Vector4<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector4<T> ret = aValue;
    ret.Round();
    return ret;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Round()
    requires NonNativeFloatingPoint<T>
{
    x.Round();
    y.Round();
    z.Round();
    w.Round();
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Round()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = round(x);
    y = round(y);
    z = round(z);
    w = round(w);
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Round()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::round(x);
    y = std::round(y);
    z = std::round(z);
    w = std::round(w);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Floor(const Vector4<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector4<T> ret = aValue;
    ret.Floor();
    return ret;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Floor()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = floor(x);
    y = floor(y);
    z = floor(z);
    w = floor(w);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Floor()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::floor(x);
    y = std::floor(y);
    z = std::floor(z);
    w = std::floor(w);
    return *this;
}

/// <summary>
/// Element wise floor() (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Floor()
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = h2floor(thisPtr[0]);
    thisPtr[1]            = h2floor(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise floor() (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Floor()
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    half2 *thisPtr = reinterpret_cast<half2 *>(this);
    thisPtr[0]     = h2floor(thisPtr[0]);
    thisPtr[1]     = h2floor(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Floor()
    requires NonNativeFloatingPoint<T>
{
    x.Floor();
    y.Floor();
    z.Floor();
    w.Floor();
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::Ceil(const Vector4<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector4<T> ret = aValue;
    ret.Ceil();
    return ret;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Ceil()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = ceil(x);
    y = ceil(y);
    z = ceil(z);
    w = ceil(w);
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::Ceil()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::ceil(x);
    y = std::ceil(y);
    z = std::ceil(z);
    w = std::ceil(w);
    return *this;
}

/// <summary>
/// Element wise ceil() (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Ceil()
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = h2ceil(thisPtr[0]);
    thisPtr[1]            = h2ceil(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise ceil() (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::Ceil()
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    half2 *thisPtr = reinterpret_cast<half2 *>(this);
    thisPtr[0]     = h2ceil(thisPtr[0]);
    thisPtr[1]     = h2ceil(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::Ceil()
    requires NonNativeFloatingPoint<T>
{
    x.Ceil();
    y.Ceil();
    z.Ceil();
    w.Ceil();
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::RoundNearest(const Vector4<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector4<T> ret = aValue;
    ret.RoundNearest();
    return ret;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::RoundNearest()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = rint(x);
    y = rint(y);
    z = rint(z);
    w = rint(w);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even <para/>
/// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::RoundNearest()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::nearbyint(x);
    y = std::nearbyint(y);
    z = std::nearbyint(z);
    w = std::nearbyint(w);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::RoundNearest()
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = h2rint(thisPtr[0]);
    thisPtr[1]            = h2rint(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::RoundNearest()
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    half2 *thisPtr = reinterpret_cast<half2 *>(this);
    thisPtr[0]     = h2rint(thisPtr[0]);
    thisPtr[1]     = h2rint(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::RoundNearest()
    requires NonNativeFloatingPoint<T>
{
    x.RoundNearest();
    y.RoundNearest();
    z.RoundNearest();
    w.RoundNearest();
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> Vector4<T>::RoundZero(const Vector4<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector4<T> ret = aValue;
    ret.RoundZero();
    return ret;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::RoundZero()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = trunc(x);
    y = trunc(y);
    z = trunc(z);
    w = trunc(w);
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
Vector4<T> &Vector4<T>::RoundZero()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::trunc(x);
    y = std::trunc(y);
    z = std::trunc(z);
    w = std::trunc(w);
    return *this;
}

/// <summary>
/// Element wise round toward zero (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::RoundZero()
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    thisPtr[0]            = h2trunc(thisPtr[0]);
    thisPtr[1]            = h2trunc(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise round toward zero (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<T> &Vector4<T>::RoundZero()
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    half2 *thisPtr = reinterpret_cast<half2 *>(this);
    thisPtr[0]     = h2trunc(thisPtr[0]);
    thisPtr[1]     = h2trunc(thisPtr[1]);
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector4<T> &Vector4<T>::RoundZero()
    requires NonNativeFloatingPoint<T>
{
    x.RoundZero();
    y.RoundZero();
    z.RoundZero();
    w.RoundZero();
    return *this;
}
#pragma endregion

#pragma region Compare per element
/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
Vector4<byte> Vector4<T>::CompareEQEps(const Vector4<T> &aLeft, const Vector4<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && HostCode<T>
{
    Vector4<T> left  = aLeft;
    Vector4<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);
    MakeNANandINFValid(left.z, right.z);
    MakeNANandINFValid(left.w, right.w);

    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(std::abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(std::abs(left.y - right.y) <= aEpsilon) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(std::abs(left.z - right.z) <= aEpsilon) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(std::abs(left.w - right.w) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector4<byte> Vector4<T>::CompareEQEps(const Vector4<T> &aLeft, const Vector4<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && DeviceCode<T>
{
    Vector4<T> left  = aLeft;
    Vector4<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);
    MakeNANandINFValid(left.z, right.z);
    MakeNANandINFValid(left.w, right.w);

    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(abs(left.y - right.y) <= aEpsilon) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(abs(left.z - right.z) <= aEpsilon) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(abs(left.w - right.w) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector4<byte> Vector4<T>::CompareEQEps(const Vector4<T> &aLeft, const Vector4<T> &aRight, T aEpsilon)
    requires Is16BitFloat<T>
{
    Vector4<T> left  = aLeft;
    Vector4<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);
    MakeNANandINFValid(left.z, right.z);
    MakeNANandINFValid(left.w, right.w);

    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(T::Abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(T::Abs(left.y - right.y) <= aEpsilon) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(T::Abs(left.z - right.z) <= aEpsilon) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(T::Abs(left.w - right.w) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector4<byte> Vector4<T>::CompareEQEps(const Vector4<T> &aLeft, const Vector4<T> &aRight,
                                                   complex_basetype_t<T> aEpsilon)
    requires ComplexFloatingPoint<T>
{
    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.x, aRight.x, aEpsilon)) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.y, aRight.y, aEpsilon)) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.z, aRight.z, aEpsilon)) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.w, aRight.w, aEpsilon)) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T> DEVICE_CODE Vector4<byte> Vector4<T>::CompareEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight)
{
    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x == aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y == aRight.y) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(aLeft.z == aRight.z) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(aLeft.w == aRight.w) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector4<byte> Vector4<T>::CompareGE(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealNumber<T>
{
    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x >= aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y >= aRight.y) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(aLeft.z >= aRight.z) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(aLeft.w >= aRight.w) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector4<byte> Vector4<T>::CompareGT(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealNumber<T>
{
    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x > aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y > aRight.y) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(aLeft.z > aRight.z) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(aLeft.w > aRight.w) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector4<byte> Vector4<T>::CompareLE(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealNumber<T>
{
    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x <= aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y <= aRight.y) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(aLeft.z <= aRight.z) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(aLeft.w <= aRight.w) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector4<byte> Vector4<T>::CompareLT(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires RealNumber<T>
{
    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x < aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y < aRight.y) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(aLeft.z < aRight.z) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(aLeft.w < aRight.w) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T> DEVICE_CODE Vector4<byte> Vector4<T>::CompareNEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight)
{
    Vector4<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x != aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y != aRight.y) * TRUE_VALUE);
    ret.z = static_cast<byte>(static_cast<int>(aLeft.z != aRight.z) * TRUE_VALUE);
    ret.w = static_cast<byte>(static_cast<int>(aLeft.w != aRight.w) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpeq4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGE(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpgeu4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGT(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpgtu4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLE(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpleu4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLT(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpltu4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareNEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpne4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpeq4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGE(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpges4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGT(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpgts4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLE(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmples4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLT(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmplts4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareNEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight)
    requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return Vector4<byte>::FromUint(__vcmpne4(aLeft, aRight));
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpeq2(leftPtr[0], rightPtr[0]), __vcmpeq2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpges2(leftPtr[0], rightPtr[0]), __vcmpges2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpgts2(leftPtr[0], rightPtr[0]), __vcmpgts2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmples2(leftPtr[0], rightPtr[0]), __vcmples2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmplts2(leftPtr[0], rightPtr[0]), __vcmplts2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareNEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpne2(leftPtr[0], rightPtr[0]), __vcmpne2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpeq2(leftPtr[0], rightPtr[0]), __vcmpeq2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpgeu2(leftPtr[0], rightPtr[0]), __vcmpgeu2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpgtu2(leftPtr[0], rightPtr[0]), __vcmpgtu2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpleu2(leftPtr[0], rightPtr[0]), __vcmpleu2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpltu2(leftPtr[0], rightPtr[0]), __vcmpltu2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareNEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    const uint *leftPtr  = reinterpret_cast<const uint *>(&aLeft);
    const uint *rightPtr = reinterpret_cast<const uint *>(&aRight);
    return Vector4<byte>::FromUint(__vcmpne2(leftPtr[0], rightPtr[0]), __vcmpne2(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    return Vector4<byte>::FromUint(__heq2_mask(leftPtr[0], rightPtr[0]), __heq2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    return Vector4<byte>::FromUint(__hge2_mask(leftPtr[0], rightPtr[0]), __hge2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    return Vector4<byte>::FromUint(__hgt2_mask(leftPtr[0], rightPtr[0]), __hgt2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    return Vector4<byte>::FromUint(__hle2_mask(leftPtr[0], rightPtr[0]), __hle2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    return Vector4<byte>::FromUint(__hlt2_mask(leftPtr[0], rightPtr[0]), __hlt2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareNEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    const nv_bfloat162 *leftPtr  = reinterpret_cast<const nv_bfloat162 *>(&aLeft);
    const nv_bfloat162 *rightPtr = reinterpret_cast<const nv_bfloat162 *>(&aRight);
    return Vector4<byte>::FromUint(__hne2_mask(leftPtr[0], rightPtr[0]), __hne2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    return Vector4<byte>::FromUint(__heq2_mask(leftPtr[0], rightPtr[0]), __heq2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    return Vector4<byte>::FromUint(__hge2_mask(leftPtr[0], rightPtr[0]), __hge2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareGT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    return Vector4<byte>::FromUint(__hgt2_mask(leftPtr[0], rightPtr[0]), __hgt2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLE(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    return Vector4<byte>::FromUint(__hle2_mask(leftPtr[0], rightPtr[0]), __hle2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareLT(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    return Vector4<byte>::FromUint(__hlt2_mask(leftPtr[0], rightPtr[0]), __hlt2_mask(leftPtr[1], rightPtr[1]));
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector4<byte> Vector4<T>::CompareNEQ(const Vector4<T> &aLeft, const Vector4<T> &aRight) // NOLINT
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    const half2 *leftPtr  = reinterpret_cast<const half2 *>(&aLeft);
    const half2 *rightPtr = reinterpret_cast<const half2 *>(&aRight);
    return Vector4<byte>::FromUint(__hne2_mask(leftPtr[0], rightPtr[0]), __hne2_mask(leftPtr[1], rightPtr[1]));
}
#pragma endregion

#pragma region Data accessors
/// <summary>
/// return sub-vector elements
/// </summary>
template <Number T> DEVICE_CODE Vector3<T> Vector4<T>::XYZ() const
{
    return Vector3<T>(x, y, z);
}

/// <summary>
/// return sub-vector elements
/// </summary>
template <Number T> DEVICE_CODE Vector3<T> Vector4<T>::YZW() const
{
    return Vector3<T>(y, z, w);
}

/// <summary>
/// return sub-vector elements
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector4<T>::XY() const
{
    return Vector2<T>(x, y);
}

/// <summary>
/// return sub-vector elements
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector4<T>::YZ() const
{
    return Vector2<T>(y, z);
}

/// <summary>
/// return sub-vector elements
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector4<T>::ZW() const
{
    return Vector2<T>(z, w);
}

/// <summary>
/// Provide a smiliar accessor to inner data as for std container
/// </summary>
template <Number T> DEVICE_CODE T *Vector4<T>::data()
{
    return &x;
}

/// <summary>
/// Provide a smiliar accessor to inner data as for std container
/// </summary>
template <Number T> DEVICE_CODE const T *Vector4<T>::data() const
{
    return &x;
}
#pragma endregion
#pragma endregion

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector4<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ", " << aVec.z << ", " << aVec.w << ')';
    return aOs;
}

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector4<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ", " << aVec.z << ", " << aVec.w << ')';
    return aOs;
}

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
std::ostream &operator<<(std::ostream &aOs, const Vector4<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ", " << static_cast<int>(aVec.y) << ", " << static_cast<int>(aVec.z)
        << ", " << static_cast<int>(aVec.w) << ')';
    return aOs;
}

template <HostCode T2>
std::wostream &operator<<(std::wostream &aOs, const Vector4<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ", " << static_cast<int>(aVec.y) << ", " << static_cast<int>(aVec.z)
        << ", " << static_cast<int>(aVec.w) << ')';
    return aOs;
}

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector4<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y >> aVec.z >> aVec.w;
    return aIs;
}

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector4<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y >> aVec.z >> aVec.w;
    return aIs;
}

template <HostCode T2>
std::istream &operator>>(std::istream &aIs, Vector4<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    aIs >> temp;
    aVec.y = static_cast<T2>(temp);
    aIs >> temp;
    aVec.z = static_cast<T2>(temp);
    aIs >> temp;
    aVec.w = static_cast<T2>(temp);
    return aIs;
}

template <HostCode T2>
std::wistream &operator>>(std::wistream &aIs, Vector4<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    aIs >> temp;
    aVec.y = static_cast<T2>(temp);
    aIs >> temp;
    aVec.z = static_cast<T2>(temp);
    aIs >> temp;
    aVec.w = static_cast<T2>(temp);
    return aIs;
}

} // namespace mpp