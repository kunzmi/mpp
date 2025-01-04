#pragma once
#include "defines.h"
#include "exception.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include "safeCast.h"
#include "vector_typetraits.h"
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
namespace opp
{
// these types are only used with CUDA, but nevertheless they need
// to be defined, so we set them to some knwon type of same size:
using nv_bfloat162 = int;
using half2        = float;
using float2       = double;
} // namespace opp

// no arguments to these intrinsics directly depend on a template parameter,
// so a declaration must be available:
opp::float2 __half22float2(opp::half2);
opp::half2 __float22half2_rn(opp::float2);
opp::float2 __bfloat1622float2(opp::nv_bfloat162);
opp::nv_bfloat162 __float22bfloat162_rn(opp::float2);
#endif

namespace opp
{

// forward declaration:
template <Number T> struct Vector1;
template <Number T> struct Vector2;
template <Number T> struct Vector3;
template <Number T> struct Vector4;

enum class Axis2D
{
    X = 0,
    Y = 1
};

inline std::ostream &operator<<(std::ostream &aOs, const Axis2D &aAxis)
{
    switch (aAxis)
    {
        case Axis2D::X:
            aOs << 'X';
            return aOs;
        case Axis2D::Y:
            aOs << 'Y';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X or Y (0 or 1).";
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const Axis2D &aAxis)
{
    switch (aAxis)
    {
        case Axis2D::X:
            aOs << 'X';
            return aOs;
        case Axis2D::Y:
            aOs << 'Y';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X or Y (0 or 1).";
    return aOs;
}

/// <summary>
/// A two T component vector. Can replace CUDA's vector2 types
/// </summary>
template <Number T> struct alignas(2 * sizeof(T)) Vector2
{
    T x;
    T y;

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    Vector2() noexcept = default;

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE Vector2(T aVal) noexcept : x(aVal), y(aVal)
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal (especially when set to 0)
    /// </summary>
    DEVICE_CODE Vector2(int aVal) noexcept
        requires(!IsInt<T>)
        : x(static_cast<T>(aVal)), y(static_cast<T>(aVal))
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1]]
    /// </summary>
    DEVICE_CODE explicit Vector2(T aVal[2]) noexcept : x(aVal[0]), y(aVal[1])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY]
    /// </summary>
    DEVICE_CODE Vector2(T aX, T aY) noexcept : x(aX), y(aY)
    {
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromUint(const uint &aUint) noexcept
        requires TwoBytesSizeType<T>
    {
        return Vector2(*reinterpret_cast<const Vector2<T> *>(&aUint));
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <Number T2> DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            Vector2<T2> temp(aVec);
            temp.template ClampToTargetType<T>();
            x = static_cast<T>(temp.x);
            y = static_cast<T>(temp.y);
        }
        else
        {
            x = static_cast<T>(aVec.x);
            y = static_cast<T>(aVec.y);
        }
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(Vector2<T2> &aVec) noexcept
        // Disable the non-const variant for half and bfloat to / from float,
        // otherwise the const specialization will never be picked up:
        requires(!(IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>) &&
                 !(IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>))
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            aVec.template ClampToTargetType<T>();
        }
        x = static_cast<T>(aVec.x);
        y = static_cast<T>(aVec.y);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>
    {
        const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
        nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
        *thisPtr              = __float22bfloat162_rn(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for BFloat2 to float2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>
    {
        const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
        float2 *thisPtr             = reinterpret_cast<float2 *>(this);
        *thisPtr                    = __bfloat1622float2(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to half2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>
    {
        const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
        half2 *thisPtr        = reinterpret_cast<half2 *>(this);
        *thisPtr              = __float22half2_rn(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for half2 to float2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>
    {
        const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
        float2 *thisPtr      = reinterpret_cast<float2 *>(this);
        *thisPtr             = __half22float2(*aVecPtr);
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromNV16BitFloat(const nv_bfloat162 &aNVBfloat2) noexcept
        requires IsBFloat16<T> && CUDA_ONLY<T>
    {
        return Vector2(*reinterpret_cast<const Vector2<T> *>(&aNVBfloat2));
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromNV16BitFloat(const half2 &aNVHalf2) noexcept
        requires IsHalfFp16<T> && CUDA_ONLY<T>
    {
        return Vector2(*reinterpret_cast<const Vector2<T> *>(&aNVHalf2));
    }

    ~Vector2() = default;

    Vector2(const Vector2 &) noexcept            = default;
    Vector2(Vector2 &&) noexcept                 = default;
    Vector2 &operator=(const Vector2 &) noexcept = default;
    Vector2 &operator=(Vector2 &&) noexcept      = default;

  private:
    // if we make those converter public we will get in trouble with some T constructors / operators
    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator const uint &() const
        requires TwoBytesSizeType<T>
    {
        return *reinterpret_cast<const uint *>(this);
    }

    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator uint &()
        requires TwoBytesSizeType<T>
    {
        return *reinterpret_cast<uint *>(this);
    }

    /// <summary>
    /// converter to nv_bfloat162 for SIMD operations
    /// </summary>
    DEVICE_CODE operator const nv_bfloat162 &() const
        requires IsBFloat16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<const nv_bfloat162 *>(this);
    }

    /// <summary>
    /// converter to nv_bfloat162 for SIMD operations
    /// </summary>
    DEVICE_CODE operator nv_bfloat162 &()
        requires IsBFloat16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<nv_bfloat162 *>(this);
    }

    /// <summary>
    /// converter to half2 for SIMD operations
    /// </summary>
    DEVICE_CODE operator const half2 &() const
        requires IsHalfFp16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<const half2 *>(this);
    }

    /// <summary>
    /// converter to half2 for SIMD operations
    /// </summary>
    DEVICE_CODE operator half2 &()
        requires IsHalfFp16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<half2 *>(this);
    }

#pragma endregion
  public:
#pragma region Operators
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector2 &) const = default;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector2 aLeft, Vector2 aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);

        bool res = std::abs(aLeft.x - aRight.x) <= aEpsilon;
        res &= std::abs(aLeft.y - aRight.y) <= aEpsilon;
        return res;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector2 aLeft, Vector2 aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);

        bool res = abs(aLeft.x - aRight.x) <= aEpsilon;
        res &= abs(aLeft.y - aRight.y) <= aEpsilon;
        return res;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector2 aLeft, Vector2 aRight, T aEpsilon)
        requires Is16BitFloat<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);

        bool res = T::Abs(aLeft.x - aRight.x) <= aEpsilon;
        res &= T::Abs(aLeft.y - aRight.y) <= aEpsilon;
        return res;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector2 &aLeft, const Vector2 &aRight,
                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>
    {
        bool res = T::EqEps(aLeft.x, aRight.x, aEpsilon);
        res &= T::EqEps(aLeft.y, aRight.y, aEpsilon);
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires RealNumber<T>
    {
        bool res = x < aOther.x;
        res &= y < aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires RealNumber<T>
    {
        bool res = x <= aOther.x;
        res &= y <= aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires RealNumber<T>
    {
        bool res = x > aOther.x;
        res &= y > aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires RealNumber<T>
    {
        bool res = x >= aOther.x;
        res &= y >= aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
    {
        bool res = x == aOther.x;
        res &= y == aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if any element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
    {
        bool res = x != aOther.x;
        res |= y != aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpgeu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpgtu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpleu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpltu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpne2(*this, aOther) != 0U;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpges2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpgts2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmples2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmplts2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpne2(*this, aOther) != 0U;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbeq2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbge2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbgt2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hble2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hblt2(*this, aOther);
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        // __hbne2 returns true only if both elements are != but we need true if any element is !=
        // so we use hbeq and negate the result
        return !(__hbeq2(*this, aOther));
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires RealSignedNumber<T> || ComplexNumber<T>
    {
        return Vector2<T>(-x, -y);
    }

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vnegss2(*this));
    }

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires Is16BitFloat<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hneg2(*this));
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(T aOther)
    {
        x += aOther;
        y += aOther;
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
    {
        x += aOther.x;
        y += aOther.y;
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vaddus2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vaddss2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hadd2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
    {
        return Vector2<T>{T(x + aOther.x), T(y + aOther.y)};
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vaddus2(*this, aOther));
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vaddss2(*this, aOther));
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hadd2(*this, aOther));
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(T aOther)
    {
        x -= aOther;
        y -= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
    {
        x -= aOther.x;
        y -= aOther.y;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vsubus2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vsubss2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hsub2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
    {
        x = aOther.x - x;
        y = aOther.y - y;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vsubus2(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vsubss2(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hsub2(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
    {
        return Vector2<T>{T(x - aOther.x), T(y - aOther.y)};
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vsubus2(*this, aOther));
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vsubss2(*this, aOther));
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hsub2(*this, aOther));
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector2 &operator*=(T aOther)
    {
        x *= aOther;
        y *= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector2 &operator*=(const Vector2 &aOther)
    {
        x *= aOther.x;
        y *= aOther.y;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator*=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hmul2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator*(const Vector2 &aOther) const
    {
        return Vector2<T>{T(x * aOther.x), T(y * aOther.y)};
    }

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator*(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hmul2(*this, aOther));
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector2 &operator/=(T aOther)
    {
        x /= aOther;
        y /= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector2 &operator/=(const Vector2 &aOther)
    {
        x /= aOther.x;
        y /= aOther.y;
        return *this;
    }

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator/=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__h2div(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInv(const Vector2 &aOther)
    {
        x = aOther.x / x;
        y = aOther.y / y;
        return *this;
    }

    /// <summary>
    /// Component wise division SIMD (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInv(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__h2div(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator/(const Vector2 &aOther) const
    {
        return Vector2<T>{T(x / aOther.x), T(y / aOther.y)};
    }

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator/(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__h2div(*this, aOther));
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis2D aAxis) const
        requires DeviceCode<T>
    {
        switch (aAxis)
        {
            case Axis2D::X:
                return x;
            case Axis2D::Y:
                return y;
        }
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis2D aAxis) const
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
    DEVICE_CODE [[nodiscard]] T &operator[](Axis2D aAxis)
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
    [[nodiscard]] T &operator[](Axis2D aAxis)
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
    template <Number T2> [[nodiscard]] static Vector2<T> DEVICE_CODE Convert(const Vector2<T2> &aVec)
    {
        return {static_cast<T>(aVec.x), static_cast<T>(aVec.y)};
    }
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector2<T> &LShift(const Vector2<T> &aOther)
        requires RealIntegral<T>
    {
        x = x << aOther.x;
        y = y << aOther.y;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> LShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x << aRight.x;
        ret.y = aLeft.y << aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector2<T> &RShift(const Vector2<T> &aOther)
        requires RealIntegral<T>
    {
        x = x >> aOther.x;
        y = y >> aOther.y;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x >> aRight.x;
        ret.y = aLeft.y >> aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector2<T> &LShift(const T &aOther)
        requires RealIntegral<T>
    {
        x = x << aOther;
        y = y << aOther;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> LShift(const Vector2<T> &aLeft, const T &aRight)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x << aRight;
        ret.y = aLeft.y << aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector2<T> &RShift(const T &aOther)
        requires RealIntegral<T>
    {
        x = x >> aOther;
        y = y >> aOther;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RShift(const Vector2<T> &aLeft, const T &aRight)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x >> aRight;
        ret.y = aLeft.y >> aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE Vector2<T> &And(const Vector2<T> &aOther)
        requires RealIntegral<T>
    {
        x = x & aOther.x;
        y = y & aOther.y;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> And(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x & aRight.x;
        ret.y = aLeft.y & aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE Vector2<T> &Or(const Vector2<T> &aOther)
        requires RealIntegral<T>
    {
        x = x | aOther.x;
        y = y | aOther.y;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Or(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x | aRight.x;
        ret.y = aLeft.y | aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE Vector2<T> &Xor(const Vector2<T> &aOther)
        requires RealIntegral<T>
    {
        x = x ^ aOther.x;
        y = y ^ aOther.y;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Xor(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x ^ aRight.x;
        ret.y = aLeft.y ^ aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE Vector2<T> &Not()
        requires RealIntegral<T>
    {
        x = ~x;
        y = ~y;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Not(const Vector2<T> &aVec)
        requires RealIntegral<T>
    {
        Vector2<T> ret;
        ret.x = ~aVec.x;
        ret.y = ~aVec.y;
        return ret;
    }
#pragma endregion

#pragma region Methods
#pragma region Exp
    /// <summary>
    /// Element wise exponential
    /// </summary>
    Vector2<T> &Exp()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::exp(x);
        y = std::exp(y);
        return *this;
    }
    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Exp()
        requires(HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T> || ComplexFloatingPoint<T>
    {
        x = T::Exp(x);
        y = T::Exp(y);
        return *this;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector2<T> &Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = exp(x);
        y = exp(y);
        return *this;
    }

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Exp()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2exp(*this));
        return *this;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = exp(aVec.x);
        ret.y = exp(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(h2exp(aVec));
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires HostCode<T> && NativeNumber<T>
    {
        Vector2<T> ret;
        ret.x = std::exp(aVec.x);
        ret.y = std::exp(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires(HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T> || ComplexFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = T::Exp(aVec.x);
        ret.y = T::Exp(aVec.y);
        return ret;
    }
#pragma endregion

#pragma region Log
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    Vector2<T> &Ln()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::log(x);
        y = std::log(y);
        return *this;
    }
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector2<T> &Ln()
        requires(HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T> || ComplexFloatingPoint<T>
    {
        x = T::Ln(x);
        y = T::Ln(y);
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector2<T> &Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = log(x);
        y = log(y);
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Ln()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2log(*this));
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = log(aVec.x);
        ret.y = log(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(h2log(aVec));
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires HostCode<T> && NativeNumber<T>
    {
        Vector2<T> ret;
        ret.x = std::log(aVec.x);
        ret.y = std::log(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires(HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T> || ComplexFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = T::Ln(aVec.x);
        ret.y = T::Ln(aVec.y);
        return ret;
    }
#pragma endregion

#pragma region Sqr
    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqr()
    {
        x = x * x;
        y = y * y;
        return *this;
    }

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqr(const Vector2<T> &aVec)
    {
        Vector2<T> ret;
        ret.x = aVec.x * aVec.x;
        ret.y = aVec.y * aVec.y;
        return ret;
    }
#pragma endregion

#pragma region Sqrt
    /// <summary>
    /// Element wise square root
    /// </summary>
    Vector2<T> &Sqrt()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::sqrt(x);
        y = std::sqrt(y);
        return *this;
    }
    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqrt()
        requires(HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T> || ComplexFloatingPoint<T>
    {
        x = T::Sqrt(x);
        y = T::Sqrt(y);
        return *this;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = sqrt(x);
        y = sqrt(y);
        return *this;
    }

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqrt()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2sqrt(*this));
        return *this;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = sqrt(aVec.x);
        ret.y = sqrt(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(h2sqrt(aVec));
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires HostCode<T> && NativeNumber<T>
    {
        Vector2<T> ret;
        ret.x = std::sqrt(aVec.x);
        ret.y = std::sqrt(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires(HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T> || ComplexFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = T::Sqrt(aVec.x);
        ret.y = T::Sqrt(aVec.y);
        return ret;
    }
#pragma endregion

#pragma region Abs
    /// <summary>
    /// Element wise absolute
    /// </summary>
    Vector2<T> &Abs()
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        x = std::abs(x);
        y = std::abs(y);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    Vector2<T> &Abs()
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        x = T::Abs(x);
        y = T::Abs(y);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector2<T> &Abs()
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        x = abs(x);
        y = abs(y);
        return *this;
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Abs()
        requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        *this = FromUint(__vabsss2(*this));
        return *this;
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Abs()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__habs2(*this));
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        Vector2<T> ret;
        ret.x = abs(aVec.x);
        ret.y = abs(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__habs2(aVec));
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        return FromUint(__vabsss2(aVec));
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        Vector2<T> ret;
        ret.x = std::abs(aVec.x);
        ret.y = std::abs(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        Vector2<T> ret;
        ret.x = T::Abs(aVec.x);
        ret.y = T::Abs(aVec.y);
        return ret;
    }
#pragma endregion

#pragma region AbsDiff
    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::abs(x - aOther.x);
        y = std::abs(y - aOther.y);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = std::abs(aLeft.x - aRight.x);
        ret.y = std::abs(aLeft.y - aRight.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        *this = FromUint(__vabsdiffs2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        *this = FromUint(__vabsdiffu2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = abs(x - aOther.x);
        y = abs(y - aOther.y);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = abs(aLeft.x - aRight.x);
        ret.y = abs(aLeft.y - aRight.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        return FromUint(__vabsdiffs2(aLeft, aRight));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        return FromUint(__vabsdiffu2(aLeft, aRight));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        x = T::Abs(x - aOther.x);
        y = T::Abs(y - aOther.y);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        Vector2<T> ret;
        ret.x = T::Abs(aLeft.x - aRight.x);
        ret.y = T::Abs(aLeft.y - aRight.y);
        return ret;
    }
#pragma endregion

#pragma region Methods for Complex types
    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE Vector2<T> &Conj()
        requires ComplexNumber<T>
    {
        x.Conj();
        y.Conj();
        return *this;
    }

    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Conj(const Vector2<T> &aValue)
        requires ComplexNumber<T>
    {
        return {T::Conj(aValue.x), T::Conj(aValue.y)};
    }

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)  per element
    /// </summary>
    DEVICE_CODE Vector2<T> &ConjMul(const Vector2<T> &aOther)
        requires ComplexNumber<T>
    {
        x.ConjMul(aOther.x);
        y.ConjMul(aOther.y);
        return *this;
    }

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> ConjMul(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires ComplexNumber<T>
    {
        return {T::ConjMul(aLeft.x, aRight.x), T::ConjMul(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Complex magnitude per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<complex_basetype_t<T>> Magnitude() const
        requires ComplexFloatingPoint<T>
    {
        Vector2<complex_basetype_t<T>> ret;
        ret.x = x.Magnitude();
        ret.y = y.Magnitude();
        return ret;
    }

    /// <summary>
    /// Complex magnitude squared per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<complex_basetype_t<T>> MagnitudeSqr() const
        requires ComplexFloatingPoint<T>
    {
        Vector2<complex_basetype_t<T>> ret;
        ret.x = x.MagnitudeSqr();
        ret.y = y.MagnitudeSqr();
        return ret;
    }

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real)) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<complex_basetype_t<T>> Angle() const
        requires ComplexFloatingPoint<T>
    {
        Vector2<complex_basetype_t<T>> ret;
        ret.x = x.Angle();
        ret.y = y.Angle();
        return ret;
    }
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector2<T> &Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = max(aMinVal, min(x, aMaxVal));
        y = max(aMinVal, min(y, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    Vector2<T> &Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        y = std::max(aMinVal, std::min(y, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector2<T> &Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        x = T::Max(aMinVal, T::Min(x, aMaxVal));
        y = T::Max(aMinVal, T::Min(y, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector2<T> &Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
        requires ComplexNumber<T>
    {
        x.Clamp(aMinVal, aMaxVal);
        y.Clamp(aMinVal, aMaxVal);
        return *this;
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector2<T> &ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>)
    {
        return Clamp(numeric_limits_conversion<T, TTarget>::lowest(), numeric_limits_conversion<T, TTarget>::max());
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector2<T> &ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>)
    {
        return *this;
    }
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = min(x, aOther.x);
        y = min(y, aOther.y);
        return *this;
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmins2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        *this = FromUint(__vminu2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hmin2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    Vector2<T> &Min(const Vector2<T> &aOther)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::min(x, aOther.x);
        y = std::min(y, aOther.y);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
    {
        x.Min(aRight.x);
        y.Min(aRight.y);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Vector2<T>{T(min(aLeft.x, aRight.x)), T(min(aLeft.y, aRight.y))};
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmins2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vminu2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hmin2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Vector2<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
    {
        return Vector2<T>{T::Min(aLeft.x, aRight.x), T::Min(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NativeNumber<T>
    {
        return min(x, y);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        return T::Min(x, y);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T> && NativeNumber<T>
    {
        return std::min({x, y});
    }
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = max(x, aOther.x);
        y = max(y, aOther.y);
        return *this;
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmaxs2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmaxu2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hmax2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    Vector2<T> &Max(const Vector2<T> &aOther)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::max(x, aOther.x);
        y = std::max(y, aOther.y);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
    {
        x.Max(aRight.x);
        y.Max(aRight.y);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Vector2<T>{T(max(aLeft.x, aRight.x)), T(max(aLeft.y, aRight.y))};
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmaxs2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmaxu2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hmax2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Vector2<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>))
    {
        return Vector2<T>{T::Max(aLeft.x, aRight.x), T::Max(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NativeNumber<T>
    {
        return max(x, y);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        return T::Max(x, y);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T> && NativeNumber<T>
    {
        return std::max({x, y});
    }
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Round(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Round()
        requires NonNativeFloatingPoint<T>
    {
        x.Round();
        y.Round();
        return *this;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector2<T> &Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = round(x);
        y = round(y);
        return *this;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    Vector2<T> &Round()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::round(x);
        y = std::round(y);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Floor(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Vector2<T> &Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = floor(x);
        y = floor(y);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    Vector2<T> &Floor()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::floor(x);
        y = std::floor(y);
        return *this;
    }

    /// <summary>
    /// Element wise floor() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Floor()
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2floor(*this));
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Floor()
        requires NonNativeFloatingPoint<T>
    {
        x.Floor();
        y.Floor();
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ceil(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Vector2<T> &Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = ceil(x);
        y = ceil(y);
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    Vector2<T> &Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::ceil(x);
        y = std::ceil(y);
        return *this;
    }

    /// <summary>
    /// Element wise ceil() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Ceil()
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2ceil(*this));
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Ceil()
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
    DEVICE_CODE [[nodiscard]] static Vector2<T> RoundNearest(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.RoundNearest();
        return ret;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector2<T> &RoundNearest()
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
    Vector2<T> &RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::nearbyint(x);
        y = std::nearbyint(y);
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundNearest()
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2rint(*this));
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundNearest()
        requires NonNativeFloatingPoint<T>
    {
        x.RoundNearest();
        y.RoundNearest();
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RoundZero(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.RoundZero();
        return ret;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector2<T> &RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = trunc(x);
        y = trunc(y);
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    Vector2<T> &RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::trunc(x);
        y = std::trunc(y);
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundZero()
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2trunc(*this));
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundZero()
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
    DEVICE_CODE [[nodiscard]] T *data()
    {
        return &x;
    }

    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T *data() const
    {
        return &x;
    }
#pragma endregion

#pragma region Compare per element
    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    [[nodiscard]] static Vector2<byte> CompareEQEps(Vector2<T> aLeft, Vector2<T> aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);

        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(std::abs(aLeft.x - aRight.x) <= aEpsilon) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(std::abs(aLeft.y - aRight.y) <= aEpsilon) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(Vector2<T> aLeft, Vector2<T> aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);

        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(abs(aLeft.x - aRight.x) <= aEpsilon) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(abs(aLeft.y - aRight.y) <= aEpsilon) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(Vector2<T> aLeft, Vector2<T> aRight, T aEpsilon)
        requires Is16BitFloat<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);

        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(T::Abs(aLeft.x - aRight.x) <= aEpsilon) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(T::Abs(aLeft.y - aRight.y) <= aEpsilon) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight,
                                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>
    {
        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.x, aRight.x, aEpsilon)) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.y, aRight.y, aEpsilon)) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x == aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y == aRight.y) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareGE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>
    {
        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x >= aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y >= aRight.y) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareGT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>
    {
        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x > aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y > aRight.y) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareLE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>
    {
        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x <= aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y <= aRight.y) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareLT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>
    {
        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x < aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y < aRight.y) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareNEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x != aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y != aRight.y) * TRUE_VALUE);
        return ret;
    }
#pragma endregion
#pragma endregion
};

template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator+(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x + aRight), static_cast<T>(aLeft.y + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator+(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft + aRight.x), static_cast<T>(aLeft + aRight.y)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator-(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x - aRight), static_cast<T>(aLeft.y - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator-(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft - aRight.x), static_cast<T>(aLeft - aRight.y)};
}

template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator*(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x * aRight), static_cast<T>(aLeft.y * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator*(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft * aRight.x), static_cast<T>(aLeft * aRight.y)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator/(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x / aRight), static_cast<T>(aLeft.y / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator/(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft / aRight.x), static_cast<T>(aLeft / aRight.y)};
}

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

} // namespace opp
