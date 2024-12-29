#pragma once
#include "complex_typetraits.h"
#include "defines.h"
#include "exception.h"
#include "limits.h"
#include "needSaturationClamp.h"
#include "safeCast.h"
#include "vector_typetraits.h"
#include <cmath>
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
namespace image
{
class Size2D;
}

template <ComplexOrNumber T> struct Vector1;
template <ComplexOrNumber T> struct Vector2;
template <ComplexOrNumber T> struct Vector3;
template <ComplexOrNumber T> struct Vector4;

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
template <ComplexOrNumber T> struct alignas(2 * sizeof(T)) Vector2
{
    T x;
    T y;

    template <typename T2> struct same_vector_size_different_type
    {
        using vector = Vector2<T2>;
    };

    template <typename T2>
    using same_vector_size_different_type_t = typename same_vector_size_different_type<T2>::vector;

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    DEVICE_CODE Vector2() noexcept
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE Vector2(T aVal) noexcept : x(aVal), y(aVal)
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1]]
    /// </summary>
    DEVICE_CODE Vector2(T aVal[2]) noexcept : x(aVal[0]), y(aVal[1])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY]
    /// </summary>
    DEVICE_CODE Vector2(T aX, T aY) noexcept : x(aX), y(aY)
    {
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
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
    template <ComplexOrNumber T2>
    DEVICE_CODE Vector2(Vector2<T2> &aVec) noexcept
        // Disable the non-const variant for half and bfloat to / from float,
        // otherwise the const specialization will never be picked up:
        requires(!(IsBFloat16<T> && isSameType<T2, float> && CUDA_ONLY<T>) &&
                 !(isSameType<T, float> && isSameType<T2, BFloat16> && CUDA_ONLY<T>) &&
                 !(IsHalfFp16<T> && isSameType<T2, float> && CUDA_ONLY<T>) &&
                 !(isSameType<T, float> && isSameType<T2, HalfFp16> && CUDA_ONLY<T>))
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
    template <ComplexOrNumber T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires IsBFloat16<T> && isSameType<T2, float> && CUDA_ONLY<T>
    {
        const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
        nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
        *thisPtr              = __float22bfloat162_rn(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for BFloat2 to float2
    /// </summary>
    template <ComplexOrNumber T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires isSameType<T, float> && isSameType<T2, BFloat16> && CUDA_ONLY<T>
    {
        const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
        float2 *thisPtr             = reinterpret_cast<float2 *>(this);
        *thisPtr                    = __bfloat1622float2(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to half2
    /// </summary>
    template <ComplexOrNumber T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires IsHalfFp16<T> && isSameType<T2, float> && CUDA_ONLY<T>
    {
        const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
        half2 *thisPtr        = reinterpret_cast<half2 *>(this);
        *thisPtr              = __float22half2_rn(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for half2 to float2
    /// </summary>
    template <ComplexOrNumber T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept
        requires isSameType<T, float> && isSameType<T2, HalfFp16> && CUDA_ONLY<T>
    {
        const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
        float2 *thisPtr      = reinterpret_cast<float2 *>(this);
        *thisPtr             = __half22float2(*aVecPtr);
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

    /*/// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromUlong(const ulong64 &aUlong) noexcept
        requires FourBytesSizeType<T>
    {
        return Vector2(*reinterpret_cast<const Vector2<T> *>(&aUlong));
    }*/

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

    ///// <summary>
    ///// converter to ulong64 for SIMD operations
    ///// </summary>
    // DEVICE_CODE operator const ulong64 &() const
    //     requires FourBytesSizeType<T>
    //{
    //     return *reinterpret_cast<const ulong64 *>(this);
    // }

    ///// <summary>
    ///// converter to ulong64 for SIMD operations
    ///// </summary>
    // DEVICE_CODE operator ulong64 &()
    //     requires FourBytesSizeType<T>
    //{
    //     return *reinterpret_cast<ulong64 *>(this);
    // }

#pragma endregion
  public:
#pragma region Operators
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector2 &) const = default;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
    {
        bool res = x < aOther.x;
        res &= y < aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
    {
        bool res = x <= aOther.x;
        res &= y <= aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
    {
        bool res = x > aOther.x;
        res &= y > aOther.y;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
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
        requires isSameType<T, ushort> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires isSameType<T, ushort> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpgeu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires isSameType<T, ushort> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpgtu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires isSameType<T, ushort> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpleu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires isSameType<T, ushort> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpltu2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires isSameType<T, ushort> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpne2(*this, aOther) != 0U;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpges2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpgts2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmples2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmplts2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpne2(*this, aOther) != 0U;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbeq2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbge2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbgt2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hble2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hblt2(*this, aOther);
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        // __hbne2 returns true only if both elements are != but we need true if any element is !=
        // so we use hbeq and negate the result
        return !(__hbeq2(*this, aOther));
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbeq2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbge2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbgt2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hble2(*this, aOther);
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hblt2(*this, aOther);
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        // __hbne2 returns true only if both elements are != but we need true if any element is !=
        // so we use hbeq and negate the result
        return !(__hbeq2(*this, aOther));
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires SignedNumber<T> || ComplexType<T>
    {
        return Vector2<T>(-x, -y);
    }

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires isSameType<T, short> && SignedNumber<T> && CUDA_ONLY<T>
    {
        return FromUint(__vnegss2(*this));
    }

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires NonNativeType<T> && SignedNumber<T> && CUDA_ONLY<T>
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
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        *this = FromUint(__vaddus2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vaddss2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        return FromUint(__vaddus2(*this, aOther));
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        return FromUint(__vaddss2(*this, aOther));
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubus2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubss2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubus2(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubss2(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        return FromUint(__vsubus2(*this, aOther));
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        return FromUint(__vsubss2(*this, aOther));
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
        requires(IsBFloat16<T> || IsHalfFp16<T>) && CUDA_ONLY<T>
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
    template <ComplexOrNumber T2> [[nodiscard]] static Vector2<T> DEVICE_CODE Convert(const Vector2<T2> &aVec)
    {
        return {static_cast<T>(aVec.x), static_cast<T>(aVec.y)};
    }
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const Vector2<T> &aOther)
        requires Integral<T>
    {
        x = x << aOther.x;
        y = y << aOther.y;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> LShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Integral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x << aRight.x;
        ret.y = aLeft.y << aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const Vector2<T> &aOther)
        requires Integral<T>
    {
        x = x >> aOther.x;
        y = y >> aOther.y;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Integral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x >> aRight.x;
        ret.y = aLeft.y >> aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const T &aOther)
        requires Integral<T>
    {
        x = x << aOther;
        y = y << aOther;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> LShift(const Vector2<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x << aRight;
        ret.y = aLeft.y << aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const T &aOther)
        requires Integral<T>
    {
        x = x >> aOther;
        y = y >> aOther;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RShift(const Vector2<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x >> aRight;
        ret.y = aLeft.y >> aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE void And(const Vector2<T> &aOther)
        requires Integral<T>
    {
        x = x & aOther.x;
        y = y & aOther.y;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> And(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Integral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x & aRight.x;
        ret.y = aLeft.y & aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE void Or(const Vector2<T> &aOther)
        requires Integral<T>
    {
        x = x | aOther.x;
        y = y | aOther.y;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Or(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Integral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x | aRight.x;
        ret.y = aLeft.y | aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE void Xor(const Vector2<T> &aOther)
        requires Integral<T>
    {
        x = x ^ aOther.x;
        y = y ^ aOther.y;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Xor(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Integral<T>
    {
        Vector2<T> ret;
        ret.x = aLeft.x ^ aRight.x;
        ret.y = aLeft.y ^ aRight.y;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE void Not()
        requires Integral<T>
    {
        x = ~x;
        y = ~y;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Not(const Vector2<T> &aVec)
        requires Integral<T>
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
    void Exp()
        requires HostCode<T> && NativeType<T>
    {
        x = std::exp(x);
        y = std::exp(y);
    }
    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_ONLY_CODE void Exp()
        requires(HostCode<T> && NonNativeType<T>) || (!EnableSIMD<T>)
    {
        x = T::Exp(x);
        y = T::Exp(y);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = exp(x);
        y = exp(y);
    }

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_CODE void Exp()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2exp(*this));
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
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(h2exp(aVec));
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires HostCode<T> && NativeType<T>
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
        requires(HostCode<T> && NonNativeType<T>) || (!EnableSIMD<T>)
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
    void Ln()
        requires HostCode<T> && NativeType<T>
    {
        x = std::log(x);
        y = std::log(y);
    }
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires(HostCode<T> && NonNativeType<T>) || (!EnableSIMD<T>)
    {
        x = T::Ln(x);
        y = T::Ln(y);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = log(x);
        y = log(y);
    }

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_CODE void Ln()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2log(*this));
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
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(h2log(aVec));
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires HostCode<T> && NativeType<T>
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
        requires(HostCode<T> && NonNativeType<T>) || (!EnableSIMD<T>)
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
    DEVICE_CODE void Sqr()
    {
        x = x * x;
        y = y * y;
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
    void Sqrt()
        requires HostCode<T> && NativeType<T>
    {
        x = std::sqrt(x);
        y = std::sqrt(y);
    }
    /// <summary>
    /// Element wise square root
    /// </summary>
    void Sqrt()
        requires(HostCode<T> && NonNativeType<T>) || (!EnableSIMD<T>)
    {
        x = T::Sqrt(x);
        y = T::Sqrt(y);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = sqrt(x);
        y = sqrt(y);
    }

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2sqrt(*this));
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
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(h2sqrt(aVec));
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires HostCode<T> && NativeType<T>
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
        requires(HostCode<T> && NonNativeType<T>) || (!EnableSIMD<T>)
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
    void Abs()
        requires HostCode<T> && SignedNumber<T> && NativeType<T>
    {
        x = std::abs(x);
        y = std::abs(y);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    void Abs()
        requires SignedNumber<T> && NonNativeType<T>
    {
        x = T::Abs(x);
        y = T::Abs(y);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires DeviceCode<T> && SignedNumber<T> && NativeType<T>
    {
        x = abs(x);
        y = abs(y);
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE void Abs()
        requires isSameType<T, short> && SignedNumber<T> && CUDA_ONLY<T> && NativeType<T> && EnableSIMD<T>
    {
        *this = FromUint(__vabsss2(*this));
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void Abs()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__habs2(*this));
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires DeviceCode<T> && SignedNumber<T> && NativeType<T>
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
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__habs2(aVec));
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires isSameType<T, short> && SignedNumber<T> && CUDA_ONLY<T> && NativeType<T> && EnableSIMD<T>
    {
        return FromUint(__vabsss2(aVec));
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires HostCode<T> && SignedNumber<T> && NativeType<T>
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
        requires NonNativeType<T>
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
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires HostCode<T> && NativeType<T>
    {
        x = std::abs(x - aOther.x);
        y = std::abs(y - aOther.y);
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        Vector2<T> ret;
        ret.x = std::abs(aLeft.x - aRight.x);
        ret.y = std::abs(aLeft.y - aRight.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T> && NativeType<T>
    {
        *this = FromUint(__vabsdiffs2(*this, aOther));
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires isSameType<T, ushort> && CUDA_ONLY<T> && NativeType<T>
    {
        *this = FromUint(__vabsdiffu2(*this, aOther));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeType<T>
    {
        x = abs(x - aOther.x);
        y = abs(y - aOther.y);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
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
        requires isSameType<T, short> && CUDA_ONLY<T> && NativeType<T>
    {
        return FromUint(__vabsdiffs2(aLeft, aRight));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, ushort> && CUDA_ONLY<T> && NativeType<T>
    {
        return FromUint(__vabsdiffu2(aLeft, aRight));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires NonNativeType<T>
    {
        x = T::Abs(x - aOther.x);
        y = T::Abs(y - aOther.y);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeType<T>
    {
        Vector2<T> ret;
        ret.x = T::Abs(aLeft.x - aRight.x);
        ret.y = T::Abs(aLeft.y - aRight.y);
        return ret;
    }
#pragma endregion

#pragma region Dot
    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] static T Dot(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires FloatingPoint<T>
    {
        return aLeft.x * aRight.x + aLeft.y * aRight.y;
    }

    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Dot(const Vector2<T> &aRight) const
        requires FloatingPoint<T>
    {
        return x * aRight.x + y * aRight.y;
    }
#pragma endregion

#pragma region Magnitude(Sqr)
    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        return sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        return std::sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires NonNativeType<T>
    {
        return T::Sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Squared vector length
    /// </summary>
    DEVICE_CODE [[nodiscard]] T MagnitudeSqr() const
        requires FloatingPoint<T>
    {
        return Dot(*this, *this);
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] double Magnitude() const
        requires Integral<T> && DeviceCode<T>
    {
        return sqrt(MagnitudeSqr());
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] double Magnitude() const
        requires Integral<T> && HostCode<T>
    {
        return std::sqrt(MagnitudeSqr());
    }

    /// <summary>
    /// Squared vector length
    /// </summary>
    [[nodiscard]] double MagnitudeSqr() const
        requires Integral<T> && HostCode<T>
    {
        double dx = to_double(x);
        double dy = to_double(y);

        return dx * dx + dy * dy;
    }
#pragma endregion

#pragma region Normalize
    /// <summary>
    /// Normalizes the vector components
    /// </summary>
    DEVICE_CODE void Normalize()
        requires FloatingPoint<T>
    {
        *this = *this / Magnitude();
    }

    /// <summary>
    /// Normalizes a vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Normalize(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Normalize();
        return ret;
    }
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeType<T>
    {
        x = max(aMinVal, min(x, aMaxVal));
        y = max(aMinVal, min(y, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeType<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        y = std::max(aMinVal, std::min(y, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires NonNativeType<T>
    {
        x = T::Max(aMinVal, T::Min(x, aMaxVal));
        y = T::Max(aMinVal, T::Min(y, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && (!IsHalfFp16<T> || !isSameType<TTarget, short>)
    {
        Clamp(T(numeric_limits<TTarget>::lowest()), T(numeric_limits<TTarget>::max()));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && IsHalfFp16<T> && isSameType<TTarget, short>
    {
        // special case for half floats: the maximum value of short is slightly larger than the closest exact
        // integer in HalfFp16, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for HalfFp16 and short, we clamp to the exact integer smaller than short::max(), i.e. 32752
        constexpr HalfFp16 maxExactShort = HalfFp16::FromUShort(0x77FF); // = 32752
        Clamp(T(numeric_limits<TTarget>::lowest()), maxExactShort);
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>)
    {
    }
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeType<T>
    {
        x = min(x, aOther.x);
        y = min(y, aOther.y);
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T> && NativeType<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmins2(*this, aOther));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires isSameType<T, ushort> && CUDA_ONLY<T> && NativeType<T> && EnableSIMD<T>
    {
        *this = FromUint(__vminu2(*this, aOther));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires NonNativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hmin2(*this, aOther));
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    void Min(const Vector2<T> &aOther)
        requires HostCode<T>
    {
        x = std::min(x, aOther.x);
        y = std::min(y, aOther.y);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aRight)
        requires NonNativeType<T>
    {
        x.Min(aRight.x);
        y.Min(aRight.y);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        return Vector2<T>{T(min(aLeft.x, aRight.x)), T(min(aLeft.y, aRight.y))};
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, short> && NativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmins2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, ushort> && NativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vminu2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hmin2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        return Vector2<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeType<T> && HostCode<T>
    {
        return Vector2<T>{T::Min(aLeft.x, aRight.x), T::Min(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NativeType<T>
    {
        return min(x, y);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NonNativeType<T>
    {
        return T::Min(x, y);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T>
    {
        return std::min({x, y});
    }
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeType<T>
    {
        x = max(x, aOther.x);
        y = max(y, aOther.y);
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires isSameType<T, short> && NativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmaxs2(*this, aOther));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires isSameType<T, ushort> && NativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmaxu2(*this, aOther));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires NonNativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hmax2(*this, aOther));
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    void Max(const Vector2<T> &aOther)
        requires HostCode<T> && NativeType<T>
    {
        x = std::max(x, aOther.x);
        y = std::max(y, aOther.y);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aRight)
        requires NonNativeType<T>
    {
        x.Max(aRight.x);
        y.Max(aRight.y);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        return Vector2<T>{T(max(aLeft.x, aRight.x)), T(max(aLeft.y, aRight.y))};
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, short> && NativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmaxs2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, ushort> && NativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmaxu2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeType<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hmax2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        return Vector2<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeType<T>
    {
        return Vector2<T>{T::Max(aLeft.x, aRight.x), T::Max(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NativeType<T>
    {
        return max(x, y);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NonNativeType<T>
    {
        return T::Max(x, y);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T>
    {
        return std::max({x, y});
    }
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Round(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<int> RoundI(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Round();
        return Vector2<int>(ret);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires FloatingComplexType<T>
    {
        x.Round();
        y.Round();
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = round(x);
        y = round(y);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    void Round()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::round(x);
        y = std::round(y);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE void Round()
        requires NonNativeType<T>
    {
        x.Round();
        y.Round();
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Floor(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<int> FloorI(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Floor();
        return Vector2<int>(ret);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE void Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = floor(x);
        y = floor(y);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    void Floor()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::floor(x);
        y = std::floor(y);
    }

    /// <summary>
    /// Element wise floor() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void Floor()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2floor(*this));
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE void Floor()
        requires NonNativeType<T> && FloatingPoint<T> && HostCode<T>
    {
        x.Floor();
        y.Floor();
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ceil(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<int> CeilI(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.Ceil();
        return Vector2<int>(ret);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE void Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = ceil(x);
        y = ceil(y);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    void Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::ceil(x);
        y = std::ceil(y);
    }

    /// <summary>
    /// Element wise ceil() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void Ceil()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2ceil(*this));
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE void Ceil()
        requires NonNativeType<T> && FloatingPoint<T> && HostCode<T>
    {
        x.Ceil();
        y.Ceil();
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RoundNearest(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.RoundNearest();
        return ret;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE void RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = __float2int_rn(x);
        y = __float2int_rn(y);
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    void RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::nearbyint(x);
        y = std::nearbyint(y);
    }

    /// <summary>
    /// Element wise round nearest ties to even (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void RoundNearest()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2rint(*this));
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE void RoundNearest()
        requires NonNativeType<T> && FloatingPoint<T> && HostCode<T>
    {
        x.RoundNearest();
        y.RoundNearest();
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RoundZero(const Vector2<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector2<T> ret = aValue;
        ret.RoundZero();
        return ret;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE void RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = __float2int_rz(x);
        y = __float2int_rz(y);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    void RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::trunc(x);
        y = std::trunc(y);
    }

    /// <summary>
    /// Element wise round toward zero (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void RoundZero()
        requires NonNativeType<T> && FloatingPoint<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2trunc(*this));
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE void RoundZero()
        requires NonNativeType<T> && FloatingPoint<T> && HostCode<T>
    {
        x.RoundZero();
        y.RoundZero();
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

#pragma region Compare
    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = byte(aLeft.x == aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y == aRight.y) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareGE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = byte(aLeft.x >= aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y >= aRight.y) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareGT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = byte(aLeft.x > aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y > aRight.y) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareLE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = byte(aLeft.x <= aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y <= aRight.y) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareLT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = byte(aLeft.x < aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y < aRight.y) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareNEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight)
    {
        Vector2<byte> ret;
        ret.x = byte(aLeft.x != aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y != aRight.y) * TRUE_VALUE;
        return ret;
    }
#pragma endregion
#pragma endregion
};

template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator+(const Vector2<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft.x + aRight), T(aLeft.y + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator+(T2 aLeft, const Vector2<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft + aRight.x), T(aLeft + aRight.y)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator-(const Vector2<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft.x - aRight), T(aLeft.y - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator-(T2 aLeft, const Vector2<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft - aRight.x), T(aLeft - aRight.y)};
}

template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator*(const Vector2<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft.x * aRight), T(aLeft.y * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator*(T2 aLeft, const Vector2<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft * aRight.x), T(aLeft * aRight.y)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator/(const Vector2<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft.x / aRight), T(aLeft.y / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator/(T2 aLeft, const Vector2<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector2<T>{T(aLeft / aRight.x), T(aLeft / aRight.y)};
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

} // namespace opp
