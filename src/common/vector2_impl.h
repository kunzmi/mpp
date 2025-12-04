#pragma once
#include "bfloat16.h"
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
#include <cmath>
#include <common/utilities.h>
#include <concepts>
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
// to be defined, so we set them to some knwon type of same size:
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
#endif

namespace mpp
{

#pragma region Constructors

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::FromUint(const uint &aUint) noexcept
    requires TwoBytesSizeType<T>
{
    return Vector2(*reinterpret_cast<const Vector2<T> *>(&aUint));
}

/// <summary>
/// Type conversion with saturation if needed<para/>
/// E.g.: when converting int to byte, values are clamped to 0..255<para/>
/// But when converting byte to int, no clamping operation is performed.
/// </summary>
template <Number T> template <Number T2> DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec) noexcept
{
    if constexpr (need_saturation_clamp_v<T2, T>)
    {
        Vector2<T2> temp(aVec);
        temp.template ClampToTargetType<T>();
        x = StaticCast<T2, T>(temp.x);
        y = StaticCast<T2, T>(temp.y);
    }
    else
    {
        x = StaticCast<T2, T>(aVec.x);
        y = StaticCast<T2, T>(aVec.y);
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
DEVICE_CODE Vector2<T>::Vector2(Vector2<T2> &aVec) noexcept
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
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to BFloat2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec) noexcept
    requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>
{
    const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    *thisPtr              = __float22bfloat162_rn(*aVecPtr);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to BFloat2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode)
    requires IsBFloat16<T> && IsFloat<T2>
{
    if constexpr (CUDA_ONLY<T>)
    {
        if (aRoundingMode == RoundingMode::NearestTiesToEven)
        {
            const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
            nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
            *thisPtr              = __float22bfloat162_rn(*aVecPtr);
        }
        else
        {
            x = BFloat16(aVec.x, aRoundingMode);
            y = BFloat16(aVec.y, aRoundingMode);
        }
    }
    else
    {
        x = BFloat16(aVec.x, aRoundingMode);
        y = BFloat16(aVec.y, aRoundingMode);
    }
}

/// <summary>
/// Type conversion using CUDA intrinsics for BFloat2 to float2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec) noexcept
    requires IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>
{
    const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
    float2 *thisPtr             = reinterpret_cast<float2 *>(this);
    *thisPtr                    = __bfloat1622float2(*aVecPtr);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to half2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec) noexcept
    requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>
{
    const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    *thisPtr              = __float22half2_rn(*aVecPtr);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float to half
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode)
    requires IsHalfFp16<T> && IsFloat<T2>
{
    if constexpr (CUDA_ONLY<T>)
    {
        if (aRoundingMode == RoundingMode::NearestTiesToEven)
        {
            const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
            half2 *thisPtr        = reinterpret_cast<half2 *>(this);
            *thisPtr              = __float22half2_rn(*aVecPtr);
        }
        else
        {
            x = HalfFp16(aVec.x, aRoundingMode);
            y = HalfFp16(aVec.y, aRoundingMode);
        }
    }
    else
    {
        x = HalfFp16(aVec.x, aRoundingMode);
        y = HalfFp16(aVec.y, aRoundingMode);
    }
}

/// <summary>
/// Type conversion using CUDA intrinsics for half2 to float2
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec) noexcept
    requires IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>
{
    const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
    float2 *thisPtr      = reinterpret_cast<float2 *>(this);
    *thisPtr             = __half22float2(*aVecPtr);
}

/// <summary>
/// Type conversion for complex with rounding (only for float to bfloat/halffloat)
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector2<T>::Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode)
    requires ComplexFloatingPoint<T> && ComplexFloatingPoint<T2> &&
                 NonNativeFloatingPoint<complex_basetype_t<remove_vector_t<T>>> &&
                 std::same_as<float, complex_basetype_t<remove_vector_t<T2>>>
    : x(T(aVec.x, aRoundingMode)), y(T(aVec.y, aRoundingMode))
{
}

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::FromNV16BitFloat(const nv_bfloat162 &aNVBfloat2) noexcept
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    return Vector2(*reinterpret_cast<const Vector2<T> *>(&aNVBfloat2));
}

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::FromNV16BitFloat(const half2 &aNVHalf2) noexcept
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    return Vector2(*reinterpret_cast<const Vector2<T> *>(&aNVHalf2));
}

// if we make those converter public we will get in trouble with some T constructors / operators
/// <summary>
/// converter to uint for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T>::operator const uint &() const
    requires TwoBytesSizeType<T>
{
    return *reinterpret_cast<const uint *>(this);
}

/// <summary>
/// converter to uint for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T>::operator uint &()
    requires TwoBytesSizeType<T>
{
    return *reinterpret_cast<uint *>(this);
}

/// <summary>
/// converter to nv_bfloat162 for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T>::operator const nv_bfloat162 &() const
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<const nv_bfloat162 *>(this);
}

/// <summary>
/// converter to nv_bfloat162 for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T>::operator nv_bfloat162 &()
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<nv_bfloat162 *>(this);
}

/// <summary>
/// converter to half2 for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T>::operator const half2 &() const
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<const half2 *>(this);
}

/// <summary>
/// converter to half2 for SIMD operations
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T>::operator half2 &()
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<half2 *>(this);
}

#pragma endregion

#pragma region Operators
// don't use space-ship operator as it returns true if any comparison returns true.
// But NPP only returns true if all channels fulfill the comparison.
// auto operator<=>(const Vector2 &) const = default;

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::EqEps(const Vector2 &aLeft, const Vector2 &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && HostCode<T>
{
    Vector2<T> left  = aLeft;
    Vector2<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);

    bool res = std::abs(left.x - right.x) <= aEpsilon;
    res &= std::abs(left.y - right.y) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::EqEps(const Vector2 &aLeft, const Vector2 &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && DeviceCode<T>
{
    Vector2<T> left  = aLeft;
    Vector2<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);

    bool res = abs(left.x - right.x) <= aEpsilon;
    res &= abs(left.y - right.y) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::EqEps(const Vector2 &aLeft, const Vector2 &aRight, T aEpsilon)
    requires Is16BitFloat<T>
{
    Vector2<T> left  = aLeft;
    Vector2<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);

    bool res = T::Abs(left.x - right.x) <= aEpsilon;
    res &= T::Abs(left.y - right.y) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::EqEps(const Vector2 &aLeft, const Vector2 &aRight, complex_basetype_t<T> aEpsilon)
    requires ComplexFloatingPoint<T>
{
    bool res = T::EqEps(aLeft.x, aRight.x, aEpsilon);
    res &= T::EqEps(aLeft.y, aRight.y, aEpsilon);
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::operator<(const Vector2 &aOther) const
    requires RealNumber<T>
{
    bool res = x < aOther.x;
    res &= y < aOther.y;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::operator<=(const Vector2 &aOther) const
    requires RealNumber<T>
{
    bool res = x <= aOther.x;
    res &= y <= aOther.y;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::operator>(const Vector2 &aOther) const
    requires RealNumber<T>
{
    bool res = x > aOther.x;
    res &= y > aOther.y;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector2<T>::operator>=(const Vector2 &aOther) const
    requires RealNumber<T>
{
    bool res = x >= aOther.x;
    res &= y >= aOther.y;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T> DEVICE_CODE bool Vector2<T>::operator==(const Vector2 &aOther) const
{
    bool res = x == aOther.x;
    res &= y == aOther.y;
    return res;
}

/// <summary>
/// Returns true if any element comparison is true
/// </summary>
template <Number T> DEVICE_CODE bool Vector2<T>::operator!=(const Vector2 &aOther) const
{
    bool res = x != aOther.x;
    res |= y != aOther.y;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator==(const Vector2 &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator>=(const Vector2 &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpgeu2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator>(const Vector2 &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpgtu2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator<=(const Vector2 &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpleu2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator<(const Vector2 &aOther) const
    requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpltu2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator!=(const Vector2 &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpne2(*this, aOther) != 0U;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator==(const Vector2 &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator>=(const Vector2 &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpges2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator>(const Vector2 &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpgts2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator<=(const Vector2 &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmples2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator<(const Vector2 &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmplts2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator!=(const Vector2 &aOther) const
    requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpne2(*this, aOther) != 0U;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator==(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __hbeq2(*this, aOther);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator>=(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __hbge2(*this, aOther);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator>(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __hbgt2(*this, aOther);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator<=(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __hble2(*this, aOther);
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator<(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __hblt2(*this, aOther);
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE bool Vector2<T>::operator!=(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    // __hbne2 returns true only if both elements are != but we need true if any element is !=
    // so we use hbeq and negate the result
    return !(__hbeq2(*this, aOther));
}

/// <summary>
/// Negation
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator-() const
    requires RealSignedNumber<T> || ComplexNumber<T>
{
    return Vector2<T>(-x, -y);
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator-() const
    requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vnegss2(*this));
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator-() const
    requires Is16BitFloat<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__hneg2(*this));
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator+=(T aOther)
{
    x += aOther;
    y += aOther;
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator+=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x += aOther;
    y += aOther;
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator+=(const Vector2 &aOther)
{
    x += aOther.x;
    y += aOther.y;
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator+=(const Vector2 &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vaddus2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator+=(const Vector2 &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vaddss2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator+=(const Vector2 &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__hadd2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector2<T>::operator+(const Vector2 &aOther) const
{
    return Vector2<T>{T(x + aOther.x), T(y + aOther.y)};
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator+(const Vector2 &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vaddus2(*this, aOther));
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator+(const Vector2 &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vaddss2(*this, aOther));
}

/// <summary>
/// Component wise addition SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator+(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__hadd2(*this, aOther));
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator-=(T aOther)
{
    x -= aOther;
    y -= aOther;
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator-=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x -= aOther;
    y -= aOther;
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator-=(const Vector2 &aOther)
{
    x -= aOther.x;
    y -= aOther.y;
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator-=(const Vector2 &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubus2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator-=(const Vector2 &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubss2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator-=(const Vector2 &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__hsub2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise subtraction (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::SubInv(const Vector2 &aOther)
{
    x = aOther.x - x;
    y = aOther.y - y;
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::SubInv(const Vector2 &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubus2(aOther, *this));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::SubInv(const Vector2 &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vsubss2(aOther, *this));
    return *this;
}

/// <summary>
/// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::SubInv(const Vector2 &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__hsub2(aOther, *this));
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector2<T>::operator-(const Vector2 &aOther) const
{
    return Vector2<T>{T(x - aOther.x), T(y - aOther.y)};
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator-(const Vector2 &aOther) const
    requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vsubus2(*this, aOther));
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator-(const Vector2 &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vsubss2(*this, aOther));
}

/// <summary>
/// Component wise subtraction SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator-(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__hsub2(*this, aOther));
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator*=(T aOther)
{
    x *= aOther;
    y *= aOther;
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator*=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x *= aOther;
    y *= aOther;
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator*=(const Vector2 &aOther)
{
    x *= aOther.x;
    y *= aOther.y;
    return *this;
}

/// <summary>
/// Component wise multiplication SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator*=(const Vector2 &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__hmul2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector2<T>::operator*(const Vector2 &aOther) const
{
    return Vector2<T>{T(x * aOther.x), T(y * aOther.y)};
}

/// <summary>
/// Component wise multiplication SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator*(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__hmul2(*this, aOther));
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator/=(T aOther)
{
    x /= aOther;
    y /= aOther;
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator/=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x /= aOther;
    y /= aOther;
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::operator/=(const Vector2 &aOther)
{
    x /= aOther.x;
    y /= aOther.y;
    return *this;
}

/// <summary>
/// Component wise division SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::operator/=(const Vector2 &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__h2div(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise division (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::DivInv(const Vector2 &aOther)
{
    x = aOther.x / x;
    y = aOther.y / y;
    return *this;
}

/// <summary>
/// Component wise division SIMD (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInv(const Vector2 &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__h2div(aOther, *this));
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector2<T>::operator/(const Vector2 &aOther) const
{
    return Vector2<T>{T(x / aOther.x), T(y / aOther.y)};
}

/// <summary>
/// Component wise division SIMD
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::operator/(const Vector2 &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__h2div(*this, aOther));
}

/// <summary>
/// Inplace integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivRound(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTiesAwayFromZero(x, aOther.x);
    y = DivRoundTiesAwayFromZero(y, aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivRoundNearest(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundNearestEven(x, aOther.x);
    y = DivRoundNearestEven(y, aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivRoundZero(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardZero(x, aOther.x);
    y = DivRoundTowardZero(y, aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivFloor(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardNegInf(x, aOther.x);
    y = DivRoundTowardNegInf(y, aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivCeil(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardPosInf(x, aOther.x);
    y = DivRoundTowardPosInf(y, aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvRound(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTiesAwayFromZero(aOther.x, x);
    y = DivRoundTiesAwayFromZero(aOther.y, y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (inverted inplace div: this =
/// aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvRoundNearest(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundNearestEven(aOther.x, x);
    y = DivRoundNearestEven(aOther.y, y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (inverted inplace div: this = aOther /
/// this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvRoundZero(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardZero(aOther.x, x);
    y = DivRoundTowardZero(aOther.y, y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvFloor(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardNegInf(aOther.x, x);
    y = DivRoundTowardNegInf(aOther.y, y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvCeil(const Vector2 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardPosInf(aOther.x, x);
    y = DivRoundTowardPosInf(aOther.y, y);
    return *this;
}

/// <summary>
/// Integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivRound(const Vector2 &aLeft, const Vector2 &aRight)
    requires RealIntegral<T>
{
    return Vector2<T>{DivRoundTiesAwayFromZero(aLeft.x, aRight.x), DivRoundTiesAwayFromZero(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivRoundNearest(const Vector2 &aLeft, const Vector2 &aRight)
    requires RealIntegral<T>
{
    return Vector2<T>{DivRoundNearestEven(aLeft.x, aRight.x), DivRoundNearestEven(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivRoundZero(const Vector2 &aLeft, const Vector2 &aRight)
    requires RealIntegral<T>
{
    return Vector2<T>{DivRoundTowardZero(aLeft.x, aRight.x), DivRoundTowardZero(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivFloor(const Vector2 &aLeft, const Vector2 &aRight)
    requires RealIntegral<T>
{
    return Vector2<T>{DivRoundTowardNegInf(aLeft.x, aRight.x), DivRoundTowardNegInf(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivCeil(const Vector2 &aLeft, const Vector2 &aRight)
    requires RealIntegral<T>
{
    return Vector2<T>{DivRoundTowardPosInf(aLeft.x, aRight.x), DivRoundTowardPosInf(aLeft.y, aRight.y)};
}

/// <summary>
/// Inplace integer division with element wise round() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleRound(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTiesAwayFromZero(x, aScale);
    y = DivScaleRoundTiesAwayFromZero(y, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleRoundNearest(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundNearestEven(x, aScale);
    y = DivScaleRoundNearestEven(y, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleRoundZero(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTowardZero(x, aScale);
    y = DivScaleRoundTowardZero(y, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleFloor(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTowardNegInf(x, aScale);
    y = DivScaleRoundTowardNegInf(y, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleCeil(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTowardPosInf(x, aScale);
    y = DivScaleRoundTowardPosInf(y, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivRound(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRound(aOther.x);
    y.DivRound(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivRoundNearest(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRoundNearest(aOther.x);
    y.DivRoundNearest(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivRoundZero(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRoundZero(aOther.x);
    y.DivRoundZero(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivFloor(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivFloor(aOther.x);
    y.DivFloor(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivCeil(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivCeil(aOther.x);
    y.DivCeil(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvRound(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRound(aOther.x);
    y.DivInvRound(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (inverted inplace div: this =
/// aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvRoundNearest(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRoundNearest(aOther.x);
    y.DivInvRoundNearest(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (inverted inplace div: this = aOther /
/// this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvRoundZero(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRoundZero(aOther.x);
    y.DivInvRoundZero(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvFloor(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvFloor(aOther.x);
    y.DivInvFloor(aOther.y);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivInvCeil(const Vector2 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvCeil(aOther.x);
    y.DivInvCeil(aOther.y);
    return *this;
}

/// <summary>
/// Integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivRound(const Vector2 &aLeft, const Vector2 &aRight)
    requires ComplexIntegral<T>
{
    return Vector2<T>{T::DivRound(aLeft.x, aRight.x), T::DivRound(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivRoundNearest(const Vector2 &aLeft, const Vector2 &aRight)
    requires ComplexIntegral<T>
{
    return Vector2<T>{T::DivRoundNearest(aLeft.x, aRight.x), T::DivRoundNearest(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivRoundZero(const Vector2 &aLeft, const Vector2 &aRight)
    requires ComplexIntegral<T>
{
    return Vector2<T>{T::DivRoundZero(aLeft.x, aRight.x), T::DivRoundZero(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivFloor(const Vector2 &aLeft, const Vector2 &aRight)
    requires ComplexIntegral<T>
{
    return Vector2<T>{T::DivFloor(aLeft.x, aRight.x), T::DivFloor(aLeft.y, aRight.y)};
}

/// <summary>
/// Integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::DivCeil(const Vector2 &aLeft, const Vector2 &aRight)
    requires ComplexIntegral<T>
{
    return Vector2<T>{T::DivCeil(aLeft.x, aRight.x), T::DivCeil(aLeft.y, aRight.y)};
}

/// <summary>
/// Inplace integer division with element wise round() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleRound(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleRound(aScale);
    y.DivScaleRound(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleRoundNearest(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleRoundNearest(aScale);
    y.DivScaleRoundNearest(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleRoundZero(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleRoundZero(aScale);
    y.DivScaleRoundZero(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleFloor(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleFloor(aScale);
    y.DivScaleFloor(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::DivScaleCeil(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleCeil(aScale);
    y.DivScaleCeil(aScale);
    return *this;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
DEVICE_CODE const T &Vector2<T>::operator[](Axis2D aAxis) const
    requires DeviceCode<T>
{
    switch (aAxis)
    {
        case Axis2D::X:
            return x;
        case Axis2D::Y:
            return y;
    }
    return x;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
const T &Vector2<T>::operator[](Axis2D aAxis) const
    requires HostCode<T>
{
    switch (aAxis)
    {
        case Axis2D::X:
            return x;
        case Axis2D::Y:
            return y;
    }

    throw INVALIDARGUMENT(aAxis, aAxis);
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
DEVICE_CODE T &Vector2<T>::operator[](Axis2D aAxis)
    requires DeviceCode<T>
{
    switch (aAxis)
    {
        case Axis2D::X:
            return x;
        case Axis2D::Y:
            return y;
    }
    return x;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
T &Vector2<T>::operator[](Axis2D aAxis)
    requires HostCode<T>
{
    switch (aAxis)
    {
        case Axis2D::X:
            return x;
        case Axis2D::Y:
            return y;
    }

    throw INVALIDARGUMENT(aAxis, aAxis);
}
#pragma endregion

#pragma region Convert Methods
/// <summary>
/// Type conversion without saturation, direct type conversion
/// </summary>
template <Number T> template <Number T2> Vector2<T> DEVICE_CODE Vector2<T>::Convert(const Vector2<T2> &aVec)
{
    return {StaticCast<T2, T>(aVec.x), StaticCast<T2, T>(aVec.y)};
}
#pragma endregion

#pragma region Integral only Methods
/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::LShift(const Vector2<T> &aOther)
    requires RealIntegral<T>
{
    x = x << aOther.x; // NOLINT
    y = y << aOther.y; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::LShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealIntegral<T>
{
    Vector2<T> ret;              // NOLINT
    ret.x = aLeft.x << aRight.x; // NOLINT
    ret.y = aLeft.y << aRight.y; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::RShift(const Vector2<T> &aOther)
    requires RealIntegral<T>
{
    x = x >> aOther.x; // NOLINT
    y = y >> aOther.y; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::RShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealIntegral<T>
{
    Vector2<T> ret;              // NOLINT
    ret.x = aLeft.x >> aRight.x; // NOLINT
    ret.y = aLeft.y >> aRight.y; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::LShift(uint aOther)
    requires RealIntegral<T>
{
    x = x << aOther; // NOLINT
    y = y << aOther; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::LShift(const Vector2<T> &aLeft, uint aRight)
    requires RealIntegral<T>
{
    Vector2<T> ret;            // NOLINT
    ret.x = aLeft.x << aRight; // NOLINT
    ret.y = aLeft.y << aRight; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::RShift(uint aOther)
    requires RealIntegral<T>
{
    x = x >> aOther; // NOLINT
    y = y >> aOther; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::RShift(const Vector2<T> &aLeft, uint aRight)
    requires RealIntegral<T>
{
    Vector2<T> ret;            // NOLINT
    ret.x = aLeft.x >> aRight; // NOLINT
    ret.y = aLeft.y >> aRight; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise And
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::And(const Vector2<T> &aOther)
    requires RealIntegral<T>
{
    x = x & aOther.x; // NOLINT
    y = y & aOther.y; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise And
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::And(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealIntegral<T>
{
    Vector2<T> ret;             // NOLINT
    ret.x = aLeft.x & aRight.x; // NOLINT
    ret.y = aLeft.y & aRight.y; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise Or
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Or(const Vector2<T> &aOther)
    requires RealIntegral<T>
{
    x = x | aOther.x; // NOLINT
    y = y | aOther.y; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise Or
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Or(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealIntegral<T>
{
    Vector2<T> ret;             // NOLINT
    ret.x = aLeft.x | aRight.x; // NOLINT
    ret.y = aLeft.y | aRight.y; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise Xor
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Xor(const Vector2<T> &aOther)
    requires RealIntegral<T>
{
    x = x ^ aOther.x; // NOLINT
    y = y ^ aOther.y; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise Xor
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Xor(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealIntegral<T>
{
    Vector2<T> ret;             // NOLINT
    ret.x = aLeft.x ^ aRight.x; // NOLINT
    ret.y = aLeft.y ^ aRight.y; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise negation
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Not()
    requires RealIntegral<T>
{
    x = ~x; // NOLINT
    y = ~y; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise negation
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Not(const Vector2<T> &aVec)
    requires RealIntegral<T>
{
    Vector2<T> ret;  // NOLINT
    ret.x = ~aVec.x; // NOLINT
    ret.y = ~aVec.y; // NOLINT
    return ret;
}
#pragma endregion

#pragma region Methods
#pragma region Exp
/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Exp()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::exp(x);
    y = std::exp(y);
    return *this;
}
/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::Exp()
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>
{
    x = T::Exp(x);
    y = T::Exp(y);
    return *this;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Exp()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = exp(x);
    y = exp(y);
    return *this;
}

/// <summary>
/// Element wise exponential (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Exp()
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(h2exp(*this));
    return *this;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Exp(const Vector2<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = exp(aVec.x);
    ret.y = exp(aVec.y);
    return ret;
}

/// <summary>
/// Element wise exponential (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Exp(const Vector2<T> &aVec)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(h2exp(aVec));
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Exp(const Vector2<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = std::exp(aVec.x);
    ret.y = std::exp(aVec.y);
    return ret;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Exp(const Vector2<T> &aVec)
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = T::Exp(aVec.x);
    ret.y = T::Exp(aVec.y);
    return ret;
}
#pragma endregion

#pragma region Log
/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Ln()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::log(x);
    y = std::log(y);
    return *this;
}
/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Ln()
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    x = T::Ln(x);
    y = T::Ln(y);
    return *this;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Ln()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = log(x);
    y = log(y);
    return *this;
}

/// <summary>
/// Element wise natural logarithm (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Ln()
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(h2log(*this));
    return *this;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Ln(const Vector2<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = log(aVec.x);
    ret.y = log(aVec.y);
    return ret;
}

/// <summary>
/// Element wise natural logarithm (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Ln(const Vector2<T> &aVec)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(h2log(aVec));
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Ln(const Vector2<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = std::log(aVec.x);
    ret.y = std::log(aVec.y);
    return ret;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Ln(const Vector2<T> &aVec)
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = T::Ln(aVec.x);
    ret.y = T::Ln(aVec.y);
    return ret;
}
#pragma endregion

#pragma region Sqr
/// <summary>
/// Element wise square
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> &Vector2<T>::Sqr()
{
    x = x * x;
    y = y * y;
    return *this;
}

/// <summary>
/// Element wise square
/// </summary>
template <Number T> DEVICE_CODE Vector2<T> Vector2<T>::Sqr(const Vector2<T> &aVec)
{
    Vector2<T> ret; // NOLINT
    ret.x = aVec.x * aVec.x;
    ret.y = aVec.y * aVec.y;
    return ret;
}
#pragma endregion

#pragma region Sqrt
/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Sqrt()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::sqrt(x);
    y = std::sqrt(y);
    return *this;
}
/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Sqrt()
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    x = T::Sqrt(x);
    y = T::Sqrt(y);
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Sqrt()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = sqrt(x);
    y = sqrt(y);
    return *this;
}

/// <summary>
/// Element wise square root (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Sqrt()
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(h2sqrt(*this));
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Sqrt(const Vector2<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = sqrt(aVec.x);
    ret.y = sqrt(aVec.y);
    return ret;
}

/// <summary>
/// Element wise square root (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Sqrt(const Vector2<T> &aVec)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(h2sqrt(aVec));
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Sqrt(const Vector2<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = std::sqrt(aVec.x);
    ret.y = std::sqrt(aVec.y);
    return ret;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Sqrt(const Vector2<T> &aVec)
    requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = T::Sqrt(aVec.x);
    ret.y = T::Sqrt(aVec.y);
    return ret;
}
#pragma endregion

#pragma region Abs
/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Abs()
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    if constexpr (sizeof(T) < 4)
    {
        // short and sbyte are computed as int
        x = static_cast<T>(std::abs(x));
        y = static_cast<T>(std::abs(y));
    }
    else
    {
        x = std::abs(x);
        y = std::abs(y);
    }
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Abs()
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    x = T::Abs(x);
    y = T::Abs(y);
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Abs()
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = abs(x);
    y = abs(y);
    return *this;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Abs()
    requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vabsss2(*this));
    return *this;
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::Abs()
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__habs2(*this));
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Abs(const Vector2<T> &aVec)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = abs(aVec.x);
    ret.y = abs(aVec.y);
    return ret;
}

/// <summary>
/// Element wise absolute (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Abs(const Vector2<T> &aVec)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__habs2(aVec));
}

/// <summary>
/// Element wise absolute  (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Abs(const Vector2<T> &aVec)
    requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    return FromUint(__vabsss2(aVec));
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Abs(const Vector2<T> &aVec)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    if constexpr (sizeof(T) < 4)
    {
        // short and sbyte are computed as int
        ret.x = static_cast<T>(std::abs(aVec.x));
        ret.y = static_cast<T>(std::abs(aVec.y));
    }
    else
    {
        ret.x = std::abs(aVec.x);
        ret.y = std::abs(aVec.y);
    }
    return ret;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Abs(const Vector2<T> &aVec)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = T::Abs(aVec.x);
    ret.y = T::Abs(aVec.y);
    return ret;
}
#pragma endregion

#pragma region AbsDiff
/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::AbsDiff(const Vector2<T> &aOther)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = std::abs(x - aOther.x);
    y = std::abs(y - aOther.y);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = std::abs(aLeft.x - aRight.x);
    ret.y = std::abs(aLeft.y - aRight.y);
    return ret;
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::AbsDiff(const Vector2<T> &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vabsdiffs2(*this, aOther));
    return *this;
}

/// <summary>
/// Element wise absolute difference (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::AbsDiff(const Vector2<T> &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vabsdiffu2(*this, aOther));
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::AbsDiff(const Vector2<T> &aOther)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = abs(x - aOther.x);
    y = abs(y - aOther.y);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = abs(aLeft.x - aRight.x);
    ret.y = abs(aLeft.y - aRight.y);
    return ret;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    return FromUint(__vabsdiffs2(aLeft, aRight));
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    return FromUint(__vabsdiffu2(aLeft, aRight));
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::AbsDiff(const Vector2<T> &aOther)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    x = T::Abs(x - aOther.x);
    y = T::Abs(y - aOther.y);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    Vector2<T> ret; // NOLINT
    ret.x = T::Abs(aLeft.x - aRight.x);
    ret.y = T::Abs(aLeft.y - aRight.y);
    return ret;
}
#pragma endregion

#pragma region Methods for Complex types
/// <summary>
/// Conjugate complex per element
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Conj()
    requires ComplexNumber<T>
{
    x.Conj();
    y.Conj();
    return *this;
}

/// <summary>
/// Conjugate complex per element
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Conj(const Vector2<T> &aValue)
    requires ComplexNumber<T>
{
    return {T::Conj(aValue.x), T::Conj(aValue.y)};
}

/// <summary>
/// Conjugate complex multiplication: this * conj(aOther)  per element
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::ConjMul(const Vector2<T> &aOther)
    requires ComplexNumber<T>
{
    x.ConjMul(aOther.x);
    y.ConjMul(aOther.y);
    return *this;
}

/// <summary>
/// Conjugate complex multiplication: aLeft * conj(aRight) per element
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::ConjMul(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires ComplexNumber<T>
{
    return {T::ConjMul(aLeft.x, aRight.x), T::ConjMul(aLeft.y, aRight.y)};
}

/// <summary>
/// Complex magnitude per element
/// </summary>
template <Number T>
DEVICE_CODE Vector2<complex_basetype_t<T>> Vector2<T>::Magnitude() const
    requires ComplexFloatingPoint<T>
{
    Vector2<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.Magnitude();
    ret.y = y.Magnitude();
    return ret;
}

/// <summary>
/// Complex magnitude squared per element
/// </summary>
template <Number T>
DEVICE_CODE Vector2<complex_basetype_t<T>> Vector2<T>::MagnitudeSqr() const
    requires ComplexFloatingPoint<T>
{
    Vector2<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.MagnitudeSqr();
    ret.y = y.MagnitudeSqr();
    return ret;
}

/// <summary>
/// Angle between real and imaginary of a complex number (atan2(image, real)) per element
/// </summary>
template <Number T>
DEVICE_CODE Vector2<complex_basetype_t<T>> Vector2<T>::Angle() const
    requires ComplexFloatingPoint<T>
{
    Vector2<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.Angle();
    ret.y = y.Angle();
    return ret;
}
#pragma endregion

#pragma region Clamp
/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Clamp(T aMinVal, T aMaxVal)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = max(aMinVal, min(x, aMaxVal));
    y = max(aMinVal, min(y, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Clamp(T aMinVal, T aMaxVal)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::max(aMinVal, std::min(x, aMaxVal));
    y = std::max(aMinVal, std::min(y, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Clamp(T aMinVal, T aMaxVal)
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    x = T::Max(aMinVal, T::Min(x, aMaxVal));
    y = T::Max(aMinVal, T::Min(y, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
    requires ComplexNumber<T>
{
    x.Clamp(aMinVal, aMaxVal);
    y.Clamp(aMinVal, aMaxVal);
    return *this;
}

/// <summary>
/// Component wise clamp to maximum value range of given target type
/// </summary>
template <Number T>
template <Number TTarget>
DEVICE_CODE Vector2<T> &Vector2<T>::ClampToTargetType() noexcept
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
DEVICE_CODE Vector2<T> &Vector2<T>::ClampToTargetType() noexcept
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
DEVICE_CODE Vector2<T> &Vector2<T>::Min(const Vector2<T> &aOther)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = min(x, aOther.x);
    y = min(y, aOther.y);
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Min(const Vector2<T> &aOther)
    requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vmins2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Min(const Vector2<T> &aOther)
    requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
{
    *this = FromUint(__vminu2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Min(const Vector2<T> &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__hmin2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Min(const Vector2<T> &aOther)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::min(x, aOther.x);
    y = std::min(y, aOther.y);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Min(const Vector2<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    x.Min(aRight.x);
    y.Min(aRight.y);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    return Vector2<T>{T(min(aLeft.x, aRight.x)), T(min(aLeft.y, aRight.y))};
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vmins2(aLeft, aRight));
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vminu2(aLeft, aRight));
}

/// <summary>
/// Component wise minimum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__hmin2(aLeft, aRight));
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
Vector2<T> Vector2<T>::Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    return Vector2<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y)};
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    return Vector2<T>{T::Min(aLeft.x, aRight.x), T::Min(aLeft.y, aRight.y)};
}

/// <summary>
/// Returns the minimum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector2<T>::Min() const
    requires DeviceCode<T> && NativeNumber<T>
{
    return min(x, y);
}

/// <summary>
/// Returns the minimum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector2<T>::Min() const
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    return T::Min(x, y);
}

/// <summary>
/// Returns the minimum component of the vector
/// </summary>
template <Number T>
T Vector2<T>::Min() const
    requires HostCode<T> && NativeNumber<T>
{
    return std::min({x, y});
}
#pragma endregion

#pragma region Max
/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Max(const Vector2<T> &aOther)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = max(x, aOther.x);
    y = max(y, aOther.y);
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Max(const Vector2<T> &aOther)
    requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vmaxs2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Max(const Vector2<T> &aOther)
    requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromUint(__vmaxu2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Max(const Vector2<T> &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(__hmax2(*this, aOther));
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Max(const Vector2<T> &aOther)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::max(x, aOther.x);
    y = std::max(y, aOther.y);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Max(const Vector2<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    x.Max(aRight.x);
    y.Max(aRight.y);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    return Vector2<T>{T(max(aLeft.x, aRight.x)), T(max(aLeft.y, aRight.y))};
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vmaxs2(aLeft, aRight));
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromUint(__vmaxu2(aLeft, aRight));
}

/// <summary>
/// Component wise maximum (SIMD)
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires NonNativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return FromNV16BitFloat(__hmax2(aLeft, aRight));
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
Vector2<T> Vector2<T>::Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    return Vector2<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y)};
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
{
    return Vector2<T>{T::Max(aLeft.x, aRight.x), T::Max(aLeft.y, aRight.y)};
}

/// <summary>
/// Returns the maximum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector2<T>::Max() const
    requires DeviceCode<T> && NativeNumber<T>
{
    return max(x, y);
}

/// <summary>
/// Returns the maximum component of the vector
/// </summary>
template <Number T>
DEVICE_CODE T Vector2<T>::Max() const
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    return T::Max(x, y);
}

/// <summary>
/// Returns the maximum component of the vector
/// </summary>
template <Number T>
T Vector2<T>::Max() const
    requires HostCode<T> && NativeNumber<T>
{
    return std::max({x, y});
}
#pragma endregion

#pragma region Round
/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Round(const Vector2<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector2<T> ret = aValue;
    ret.Round();
    return ret;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::Round()
    requires NonNativeFloatingPoint<T>
{
    x.Round();
    y.Round();
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Round()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = round(x);
    y = round(y);
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Round()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::round(x);
    y = std::round(y);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Floor(const Vector2<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector2<T> ret = aValue;
    ret.Floor();
    return ret;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Floor()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = floor(x);
    y = floor(y);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Floor()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::floor(x);
    y = std::floor(y);
    return *this;
}

/// <summary>
/// Element wise floor() (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::Floor()
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(h2floor(*this));
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::Floor()
    requires NonNativeFloatingPoint<T>
{
    x.Floor();
    y.Floor();
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::Ceil(const Vector2<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector2<T> ret = aValue;
    ret.Ceil();
    return ret;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::Ceil()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = ceil(x);
    y = ceil(y);
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::Ceil()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::ceil(x);
    y = std::ceil(y);
    return *this;
}

/// <summary>
/// Element wise ceil() (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::Ceil()
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(h2ceil(*this));
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::Ceil()
    requires NonNativeFloatingPoint<T>
{
    x.Ceil();
    y.Ceil();
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::RoundNearest(const Vector2<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector2<T> ret = aValue;
    ret.RoundNearest();
    return ret;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::RoundNearest()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = rint(x);
    y = rint(y);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even<para/>
/// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::RoundNearest()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::nearbyint(x);
    y = std::nearbyint(y);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::RoundNearest()
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(h2rint(*this));
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::RoundNearest()
    requires NonNativeFloatingPoint<T>
{
    x.RoundNearest();
    y.RoundNearest();
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> Vector2<T>::RoundZero(const Vector2<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector2<T> ret = aValue;
    ret.RoundZero();
    return ret;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector2<T> &Vector2<T>::RoundZero()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = trunc(x);
    y = trunc(y);
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
Vector2<T> &Vector2<T>::RoundZero()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::trunc(x);
    y = std::trunc(y);
    return *this;
}

/// <summary>
/// Element wise round toward zero (SIMD)
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::RoundZero()
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(h2trunc(*this));
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector2<T> &Vector2<T>::RoundZero()
    requires NonNativeFloatingPoint<T>
{
    x.RoundZero();
    y.RoundZero();
    return *this;
}
#pragma endregion

#pragma region Data accessors
/// <summary>
/// Provide a smiliar accessor to inner data as for std container
/// </summary>
template <Number T> DEVICE_CODE T *Vector2<T>::data()
{
    return &x;
}

/// <summary>
/// Provide a smiliar accessor to inner data as for std container
/// </summary>
template <Number T> DEVICE_CODE const T *Vector2<T>::data() const
{
    return &x;
}
#pragma endregion

#pragma region Compare per element
/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
Vector2<byte> Vector2<T>::CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && HostCode<T>
{
    Vector2<T> left  = aLeft;
    Vector2<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);

    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(std::abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(std::abs(left.y - right.y) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector2<byte> Vector2<T>::CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && DeviceCode<T>
{
    Vector2<T> left  = aLeft;
    Vector2<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);

    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(abs(left.y - right.y) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector2<byte> Vector2<T>::CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight, T aEpsilon)
    requires Is16BitFloat<T>
{
    Vector2<T> left  = aLeft;
    Vector2<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);
    MakeNANandINFValid(left.y, right.y);

    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(T::Abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(T::Abs(left.y - right.y) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector2<byte> Vector2<T>::CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight,
                                                   complex_basetype_t<T> aEpsilon)
    requires ComplexFloatingPoint<T>
{
    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.x, aRight.x, aEpsilon)) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.y, aRight.y, aEpsilon)) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T> DEVICE_CODE Vector2<byte> Vector2<T>::CompareEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight)
{
    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x == aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y == aRight.y) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector2<byte> Vector2<T>::CompareGE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealNumber<T>
{
    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x >= aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y >= aRight.y) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector2<byte> Vector2<T>::CompareGT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealNumber<T>
{
    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x > aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y > aRight.y) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector2<byte> Vector2<T>::CompareLE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealNumber<T>
{
    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x <= aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y <= aRight.y) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector2<byte> Vector2<T>::CompareLT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    requires RealNumber<T>
{
    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x < aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y < aRight.y) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T> DEVICE_CODE Vector2<byte> Vector2<T>::CompareNEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight)
{
    Vector2<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x != aRight.x) * TRUE_VALUE);
    ret.y = static_cast<byte>(static_cast<int>(aLeft.y != aRight.y) * TRUE_VALUE);
    return ret;
}
#pragma endregion
#pragma endregion

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector2<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ')';
    return aOs;
}

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector2<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ')';
    return aOs;
}

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
std::ostream &operator<<(std::ostream &aOs, const Vector2<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ", " << static_cast<int>(aVec.y) << ')';
    return aOs;
}

template <HostCode T2>
std::wostream &operator<<(std::wostream &aOs, const Vector2<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ", " << static_cast<int>(aVec.y) << ')';
    return aOs;
}

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector2<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y;
    return aIs;
}

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector2<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y;
    return aIs;
}

template <HostCode T2>
std::istream &operator>>(std::istream &aIs, Vector2<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    aIs >> temp;
    aVec.y = static_cast<T2>(temp);
    return aIs;
}

template <HostCode T2>
std::wistream &operator>>(std::wistream &aIs, Vector2<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    aIs >> temp;
    aVec.y = static_cast<T2>(temp);
    return aIs;
}

} // namespace mpp