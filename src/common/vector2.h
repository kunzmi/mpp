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
    template <ComplexOrNumber T2> DEVICE_CODE Vector2(Vector2<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            aVec.template ClampToTargetType<T>();
        }
        x = static_cast<T>(aVec.x);
        y = static_cast<T>(aVec.y);
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromUint(const uint &aUint) noexcept
        requires TwoBytesSizeType<T>
    {
        return Vector2(*reinterpret_cast<const Vector2<T> *>(&aUint));
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

  public:
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
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        return FromUint(__vnegss2(*this));
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
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubus2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubss2(*this, aOther));
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
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator*(const Vector2 &aOther) const
    {
        return Vector2<T>{T(x * aOther.x), T(y * aOther.y)};
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
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator/(const Vector2 &aOther) const
    {
        return Vector2<T>{T(x / aOther.x), T(y / aOther.y)};
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

    /// <summary>
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <ComplexOrNumber T2> [[nodiscard]] static Vector2<T> DEVICE_CODE Convert(const Vector2<T2> &aVec)
    {
        return {static_cast<T>(aVec.x), static_cast<T>(aVec.y)};
    }

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

    /// <summary>
    /// Element wise exponential
    /// </summary>
    void Exp()
        requires HostCode<T>
    {
        x = std::exp(x);
        y = std::exp(y);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = exp(x);
        y = exp(y);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = exp(aVec.x);
        ret.y = exp(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires HostCode<T>
    {
        Vector2<T> ret;
        ret.x = std::exp(aVec.x);
        ret.y = std::exp(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    void Ln()
        requires HostCode<T>
    {
        x = std::log(x);
        y = std::log(y);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = log(x);
        y = log(y);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = log(aVec.x);
        ret.y = log(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires HostCode<T>
    {
        Vector2<T> ret;
        ret.x = std::log(aVec.x);
        ret.y = std::log(aVec.y);
        return ret;
    }

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

    /// <summary>
    /// Element wise square root
    /// </summary>
    void Sqrt()
        requires HostCode<T>
    {
        x = std::sqrt(x);
        y = std::sqrt(y);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = sqrt(x);
        y = sqrt(y);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector2<T> ret;
        ret.x = sqrt(aVec.x);
        ret.y = sqrt(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires HostCode<T>
    {
        Vector2<T> ret;
        ret.x = std::sqrt(aVec.x);
        ret.y = std::sqrt(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    void Abs()
        requires HostCode<T>
    {
        x = std::abs(x);
        y = std::abs(y);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires DeviceCode<T>
    {
        x = abs(x);
        y = abs(y);
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE void Abs()
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vabsss2(*this));
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires DeviceCode<T>
    {
        Vector2<T> ret;
        ret.x = abs(aVec.x);
        ret.y = abs(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        return FromUint(__vabsss2(aVec));
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires HostCode<T>
    {
        Vector2<T> ret;
        ret.x = std::abs(aVec.x);
        ret.y = std::abs(aVec.y);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires HostCode<T>
    {
        x = std::abs(x - aOther.x);
        y = std::abs(y - aOther.y);
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T>
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
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vabsdiffs2(*this, aOther));
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        *this = FromUint(__vabsdiffu2(*this, aOther));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector2<T> &aOther)
        requires DeviceCode<T>
    {
        x = abs(x - aOther.x);
        y = abs(y - aOther.y);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T>
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
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        return FromUint(__vabsdiffs2(aLeft, aRight));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        return FromUint(__vabsdiffu2(aLeft, aRight));
    }

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

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && FloatingPoint<T>
    {
        return sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && FloatingPoint<T>
    {
        return std::sqrt(Dot(*this, *this));
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

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T>
    {
        x = max(aMinVal, min(x, aMaxVal));
        y = max(aMinVal, min(y, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        y = std::max(aMinVal, std::min(y, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>)
    {
        Clamp(T(numeric_limits<TTarget>::lowest()), T(numeric_limits<TTarget>::max()));
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

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires DeviceCode<T>
    {
        x = min(x, aOther.x);
        y = min(y, aOther.y);
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vmins2(*this, aOther));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        *this = FromUint(__vminu2(*this, aOther));
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
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires DeviceCode<T>
    {
        x = max(x, aOther.x);
        y = max(y, aOther.y);
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        *this = FromUint(__vmaxs2(*this, aOther));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        *this = FromUint(__vmaxu2(*this, aOther));
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    void Max(const Vector2<T> &aOther)
        requires HostCode<T>
    {
        x = std::max(x, aOther.x);
        y = std::max(y, aOther.y);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T>
    {
        return Vector2<T>{min(aLeft.x, aRight.x), min(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        return FromUint(__vmins2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        return FromUint(__vminu2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T>
    {
        return Vector2<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T>
    {
        return Vector2<T>{max(aLeft.x, aRight.x), max(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, short> && CUDA_ONLY<T>
    {
        return FromUint(__vmaxs2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires isSameType<T, ushort> && CUDA_ONLY<T>
    {
        return FromUint(__vmaxu2(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T>
    {
        return Vector2<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y)};
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T>
    {
        return min(x, y);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T>
    {
        return std::min({x, y});
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T>
    {
        return max(x, y);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T>
    {
        return std::max({x, y});
    }

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
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = round(x);
        y = round(y);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    void Round()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::round(x);
        y = std::round(y);
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
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = floor(x);
        y = floor(y);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    void Floor()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::floor(x);
        y = std::floor(y);
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
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = ceil(x);
        y = ceil(y);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    void Ceil()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::ceil(x);
        y = std::ceil(y);
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
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = __float2int_rn(x);
        y = __float2int_rn(y);
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    void RoundNearest()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::nearbyint(x);
        y = std::nearbyint(y);
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
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = __float2int_rz(x);
        y = __float2int_rz(y);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    void RoundZero()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::trunc(x);
        y = std::trunc(y);
    }

    /// <summary>
    /// Provide a smiliar accessor to inner data similar to std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] T *data()
    {
        return &x;
    }

    /// <summary>
    /// Provide a smiliar accessor to inner data similar to std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T *data() const
    {
        return &x;
    }
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
