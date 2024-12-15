#pragma once
#include "complex_typetraits.h"
#include "defines.h"
#include "exception.h"
#include "limits.h"
#include "needSaturationClamp.h"
#include "safeCast.h"
#include "vector_typetraits.h"
#include "vector3.h" //for additional constructor from Vector3<T>
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>

namespace opp
{

// forward declaration:
template <ComplexOrNumber T> struct Vector1;
template <ComplexOrNumber T> struct Vector2;
template <ComplexOrNumber T> struct Vector3;
template <ComplexOrNumber T> struct Vector4;
template <ComplexOrNumber T> struct Vector4A;

enum class Axis4D
{
    X = 0,
    Y = 1,
    Z = 2,
    W = 3
};

inline std::ostream &operator<<(std::ostream &aOs, const Axis4D &aAxis)
{
    switch (aAxis)
    {
        case Axis4D::X:
            aOs << 'X';
            return aOs;
        case Axis4D::Y:
            aOs << 'Y';
            return aOs;
        case Axis4D::Z:
            aOs << 'Z';
            return aOs;
        case Axis4D::W:
            aOs << 'W';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y, Z or W (0, 1, 2 or 3).";
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const Axis4D &aAxis)
{
    switch (aAxis)
    {
        case Axis4D::X:
            aOs << 'X';
            return aOs;
        case Axis4D::Y:
            aOs << 'Y';
            return aOs;
        case Axis4D::Z:
            aOs << 'Z';
            return aOs;
        case Axis4D::W:
            aOs << 'W';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y, Z or W (0, 1, 2 or 3).";
    return aOs;
}

/// <summary>
/// A four T component vector. Can replace CUDA's vector4 types
/// </summary>
template <ComplexOrNumber T> struct alignas(4 * sizeof(T)) Vector4
{
    T x;
    T y;
    T z;
    T w;

    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    DEVICE_CODE Vector4() noexcept
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE Vector4(T aVal) noexcept : x(aVal), y(aVal), z(aVal), w(aVal)
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1], aVal[2], aVal[3]]
    /// </summary>
    DEVICE_CODE Vector4(T aVal[4]) noexcept : x(aVal[0]), y(aVal[1]), z(aVal[2]), w(aVal[3])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY, aZ, aW]
    /// </summary>
    DEVICE_CODE Vector4(T aX, T aY, T aZ, T aW) noexcept : x(aX), y(aY), z(aZ), w(aW)
    {
    }

    /// <summary>
    /// Usefull constructor if we want a Vector4 from 3 channel pixel Vector3 and one alpha channel
    /// </summary>
    DEVICE_CODE Vector4(const Vector3<T> &aVec3, T aAlpha) noexcept : x(aVec3.x), y(aVec3.y), z(aVec3.z), w(aAlpha)
    {
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector4 FromUint(const uint &aUint) noexcept
        requires ByteSizeType<T>
    {
        return Vector4(*reinterpret_cast<const Vector4<T> *>(&aUint));
    }

    /*/// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector4 FromUlong(const ulong64 &aUlong) noexcept
        requires TwoBytesSizeType<T>
    {
        return Vector4(*reinterpret_cast<const Vector4<T> *>(&aUlong));
    }*/

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE Vector4(const Vector4<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            Vector4<T2> temp(aVec);
            temp.template ClampToTargetType<T>();
            x = static_cast<T>(temp.x);
            y = static_cast<T>(temp.y);
            z = static_cast<T>(temp.z);
            w = static_cast<T>(temp.w);
        }
        else
        {
            x = static_cast<T>(aVec.x);
            y = static_cast<T>(aVec.y);
            z = static_cast<T>(aVec.z);
            w = static_cast<T>(aVec.w);
        }
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE Vector4(Vector4<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            aVec.template ClampToTargetType<T>();
        }
        x = static_cast<T>(aVec.x);
        y = static_cast<T>(aVec.y);
        z = static_cast<T>(aVec.z);
        w = static_cast<T>(aVec.w);
    }

    ~Vector4() = default;

    Vector4(const Vector4 &) noexcept            = default;
    Vector4(Vector4 &&) noexcept                 = default;
    Vector4 &operator=(const Vector4 &) noexcept = default;
    Vector4 &operator=(Vector4 &&) noexcept      = default;

    // implemented in Vector4A.h to avoid cyclic includes:
    Vector4 &operator=(const Vector4A<T> &) noexcept;

  private:
    // if we make those converter public we will get in trouble with some T constructors / operators
    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator const uint &() const
        requires ByteSizeType<T>
    {
        return *reinterpret_cast<const uint *>(this);
    }

    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator uint &()
        requires ByteSizeType<T>
    {
        return *reinterpret_cast<uint *>(this);
    }

    ///// <summary>
    ///// converter to ulong64 for SIMD operations
    ///// </summary>
    // DEVICE_CODE operator const ulong64 &() const
    //     requires TwoBytesSizeType<T>
    //{
    //     return *reinterpret_cast<const ulong64 *>(this);
    // }

    ///// <summary>
    ///// converter to ulong64 for SIMD operations
    ///// </summary>
    // DEVICE_CODE operator ulong64 &()
    //     requires TwoBytesSizeType<T>
    //{
    //     return *reinterpret_cast<ulong64 *>(this);
    // }

  public:
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector4 &) const = default;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector4 &aOther) const
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
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector4 &aOther) const
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
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector4 &aOther) const
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
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector4 &aOther) const
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
    DEVICE_CODE [[nodiscard]] bool operator==(const Vector4 &aOther) const
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
    DEVICE_CODE [[nodiscard]] bool operator!=(const Vector4 &aOther) const
    {
        bool res = x != aOther.x;
        res |= y != aOther.y;
        res |= z != aOther.z;
        res |= w != aOther.w;
        return res;
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator-() const
        requires SignedNumber<T> || ComplexType<T>
    {
        return Vector4<T>(-x, -y, -z, -w);
    }

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator-() const
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        return FromUint(__vnegss4(*this));
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector4 &operator+=(T aOther)
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
    DEVICE_CODE Vector4 &operator+=(const Vector4 &aOther)
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
    DEVICE_CODE Vector4 &operator+=(const Vector4 &aOther)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vaddus4(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector4 &operator+=(const Vector4 &aOther)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vaddss4(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator+(const Vector4 &aOther) const
    {
        return Vector4<T>{T(x + aOther.x), T(y + aOther.y), T(z + aOther.z), T(w + aOther.w)};
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator+(const Vector4 &aOther) const
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        return FromUint(__vaddus4(*this, aOther));
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator+(const Vector4 &aOther) const
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        return FromUint(__vaddss4(*this, aOther));
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector4 &operator-=(T aOther)
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
    DEVICE_CODE Vector4 &operator-=(const Vector4 &aOther)
    {
        x -= aOther.x;
        y -= aOther.y;
        z -= aOther.z;
        w -= aOther.w;
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector4 &operator-=(const Vector4 &aOther)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubus4(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector4 &operator-=(const Vector4 &aOther)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubss4(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator-(const Vector4 &aOther) const
    {
        return Vector4<T>{T(x - aOther.x), T(y - aOther.y), T(z - aOther.z), T(w - aOther.w)};
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator-(const Vector4 &aOther) const
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        return FromUint(__vsubus4(*this, aOther));
    }

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator-(const Vector4 &aOther) const
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        return FromUint(__vsubss4(*this, aOther));
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector4 &operator*=(T aOther)
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
    DEVICE_CODE Vector4 &operator*=(const Vector4 &aOther)
    {
        x *= aOther.x;
        y *= aOther.y;
        z *= aOther.z;
        w *= aOther.w;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator*(const Vector4 &aOther) const
    {
        return Vector4<T>{T(x * aOther.x), T(y * aOther.y), T(z * aOther.z), T(w * aOther.w)};
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector4 &operator/=(T aOther)
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
    DEVICE_CODE Vector4 &operator/=(const Vector4 &aOther)
    {
        x /= aOther.x;
        y /= aOther.y;
        z /= aOther.z;
        w /= aOther.w;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4 operator/(const Vector4 &aOther) const
    {
        return Vector4<T>{T(x / aOther.x), T(y / aOther.y), T(z / aOther.z), T(w / aOther.w)};
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis4D aAxis) const
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
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis4D aAxis) const
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
    DEVICE_CODE [[nodiscard]] T &operator[](Axis4D aAxis)
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
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] T &operator[](Axis4D aAxis)
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
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <ComplexOrNumber T2> [[nodiscard]] static Vector4<T> DEVICE_CODE Convert(const Vector4<T2> &aVec)
    {
        return {static_cast<T>(aVec.x), static_cast<T>(aVec.y), static_cast<T>(aVec.z), static_cast<T>(aVec.w)};
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const Vector4<T> &aOther)
        requires Integral<T>
    {
        x = x << aOther.x;
        y = y << aOther.y;
        z = z << aOther.z;
        w = w << aOther.w;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> LShift(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = aLeft.x << aRight.x;
        ret.y = aLeft.y << aRight.y;
        ret.z = aLeft.z << aRight.z;
        ret.w = aLeft.w << aRight.w;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const Vector4<T> &aOther)
        requires Integral<T>
    {
        x = x >> aOther.x;
        y = y >> aOther.y;
        z = z >> aOther.z;
        w = w >> aOther.w;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> RShift(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = aLeft.x >> aRight.x;
        ret.y = aLeft.y >> aRight.y;
        ret.z = aLeft.z >> aRight.z;
        ret.w = aLeft.w >> aRight.w;
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
        z = z << aOther;
        w = w << aOther;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> LShift(const Vector4<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = aLeft.x << aRight;
        ret.y = aLeft.y << aRight;
        ret.z = aLeft.z << aRight;
        ret.w = aLeft.w << aRight;
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
        z = z >> aOther;
        w = w >> aOther;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> RShift(const Vector4<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = aLeft.x >> aRight;
        ret.y = aLeft.y >> aRight;
        ret.z = aLeft.z >> aRight;
        ret.w = aLeft.w >> aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE void And(const Vector4<T> &aOther)
        requires Integral<T>
    {
        x = x & aOther.x;
        y = y & aOther.y;
        z = z & aOther.z;
        w = w & aOther.w;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> And(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = aLeft.x & aRight.x;
        ret.y = aLeft.y & aRight.y;
        ret.z = aLeft.z & aRight.z;
        ret.w = aLeft.w & aRight.w;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE void Or(const Vector4<T> &aOther)
        requires Integral<T>
    {
        x = x | aOther.x;
        y = y | aOther.y;
        z = z | aOther.z;
        w = w | aOther.w;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Or(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = aLeft.x | aRight.x;
        ret.y = aLeft.y | aRight.y;
        ret.z = aLeft.z | aRight.z;
        ret.w = aLeft.w | aRight.w;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE void Xor(const Vector4<T> &aOther)
        requires Integral<T>
    {
        x = x ^ aOther.x;
        y = y ^ aOther.y;
        z = z ^ aOther.z;
        w = w ^ aOther.w;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Xor(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = aLeft.x ^ aRight.x;
        ret.y = aLeft.y ^ aRight.y;
        ret.z = aLeft.z ^ aRight.z;
        ret.w = aLeft.w ^ aRight.w;
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
        z = ~z;
        w = ~w;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Not(const Vector4<T> &aVec)
        requires Integral<T>
    {
        Vector4<T> ret;
        ret.x = ~aVec.x;
        ret.y = ~aVec.y;
        ret.z = ~aVec.z;
        ret.w = ~aVec.w;
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
        z = std::exp(z);
        w = std::exp(w);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = exp(x);
        y = exp(y);
        z = exp(z);
        w = exp(w);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Exp(const Vector4<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector4<T> ret;
        ret.x = exp(aVec.x);
        ret.y = exp(aVec.y);
        ret.z = exp(aVec.z);
        ret.w = exp(aVec.w);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Exp(const Vector4<T> &aVec)
        requires HostCode<T>
    {
        Vector4<T> ret;
        ret.x = std::exp(aVec.x);
        ret.y = std::exp(aVec.y);
        ret.z = std::exp(aVec.z);
        ret.w = std::exp(aVec.w);
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
        z = std::log(z);
        w = std::log(w);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = log(x);
        y = log(y);
        z = log(z);
        w = log(w);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Ln(const Vector4<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector4<T> ret;
        ret.x = log(aVec.x);
        ret.y = log(aVec.y);
        ret.z = log(aVec.z);
        ret.w = log(aVec.w);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Ln(const Vector4<T> &aVec)
        requires HostCode<T>
    {
        Vector4<T> ret;
        ret.x = std::log(aVec.x);
        ret.y = std::log(aVec.y);
        ret.z = std::log(aVec.z);
        ret.w = std::log(aVec.w);
        return ret;
    }

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE void Sqr()
    {
        x = x * x;
        y = y * y;
        z = z * z;
        w = w * w;
    }

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Sqr(const Vector4<T> &aVec)
    {
        Vector4<T> ret;
        ret.x = aVec.x * aVec.x;
        ret.y = aVec.y * aVec.y;
        ret.z = aVec.z * aVec.z;
        ret.w = aVec.w * aVec.w;
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
        z = std::sqrt(z);
        w = std::sqrt(w);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = sqrt(x);
        y = sqrt(y);
        z = sqrt(z);
        w = sqrt(w);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Sqrt(const Vector4<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector4<T> ret;
        ret.x = sqrt(aVec.x);
        ret.y = sqrt(aVec.y);
        ret.z = sqrt(aVec.z);
        ret.w = sqrt(aVec.w);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Sqrt(const Vector4<T> &aVec)
        requires HostCode<T>
    {
        Vector4<T> ret;
        ret.x = std::sqrt(aVec.x);
        ret.y = std::sqrt(aVec.y);
        ret.z = std::sqrt(aVec.z);
        ret.w = std::sqrt(aVec.w);
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
        z = std::abs(z);
        w = std::abs(w);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires DeviceCode<T>
    {
        x = abs(x);
        y = abs(y);
        z = abs(z);
        w = abs(w);
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE void Abs()
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vabsss4(*this));
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Abs(const Vector4<T> &aVec)
        requires DeviceCode<T>
    {
        Vector4<T> ret;
        ret.x = abs(aVec.x);
        ret.y = abs(aVec.y);
        ret.z = abs(aVec.z);
        ret.w = abs(aVec.w);
        return ret;
    }

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Abs(const Vector4<T> &aVec)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        return FromUint(__vabsss4(aVec));
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Abs(const Vector4<T> &aVec)
        requires HostCode<T>
    {
        Vector4<T> ret;
        ret.x = std::abs(aVec.x);
        ret.y = std::abs(aVec.y);
        ret.z = std::abs(aVec.z);
        ret.w = std::abs(aVec.w);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector4<T> &aOther)
        requires HostCode<T>
    {
        x = std::abs(x - aOther.x);
        y = std::abs(y - aOther.y);
        z = std::abs(z - aOther.z);
        w = std::abs(w - aOther.w);
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires HostCode<T>
    {
        Vector4<T> ret;
        ret.x = std::abs(aLeft.x - aRight.x);
        ret.y = std::abs(aLeft.y - aRight.y);
        ret.z = std::abs(aLeft.z - aRight.z);
        ret.w = std::abs(aLeft.w - aRight.w);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector4<T> &aOther)
        requires DeviceCode<T>
    {
        x = abs(x - aOther.x);
        y = abs(y - aOther.y);
        z = abs(z - aOther.z);
        w = abs(w - aOther.w);
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector4<T> &aOther)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vabsdiffs4(*this, aOther));
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector4<T> &aOther)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vabsdiffu4(*this, aOther));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires DeviceCode<T>
    {
        Vector4<T> ret;
        ret.x = abs(aLeft.x - aRight.x);
        ret.y = abs(aLeft.y - aRight.y);
        ret.z = abs(aLeft.z - aRight.z);
        ret.w = abs(aLeft.w - aRight.w);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        return FromUint(__vabsdiffs4(aLeft, aRight));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> AbsDiff(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        return FromUint(__vabsdiffu4(aLeft, aRight));
    }

    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] static T Dot(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires FloatingPoint<T>
    {
        return aLeft.x * aRight.x + aLeft.y * aRight.y + aLeft.z * aRight.z + aLeft.w * aRight.w;
    }

    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Dot(const Vector4<T> &aRight) const
        requires FloatingPoint<T>
    {
        return x * aRight.x + y * aRight.y + z * aRight.z + w * aRight.w;
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
        double dz = to_double(z);
        double dw = to_double(w);

        return dx * dx + dy * dy + dz * dz + dw * dw;
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
    DEVICE_CODE [[nodiscard]] static Vector4<T> Normalize(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
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
        z = max(aMinVal, min(z, aMaxVal));
        w = max(aMinVal, min(w, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        y = std::max(aMinVal, std::min(y, aMaxVal));
        z = std::max(aMinVal, std::min(z, aMaxVal));
        w = std::max(aMinVal, std::min(w, aMaxVal));
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
    DEVICE_CODE void Min(const Vector4<T> &aRight)
        requires DeviceCode<T>
    {
        x = min(x, aRight.x);
        y = min(y, aRight.y);
        z = min(z, aRight.z);
        w = min(w, aRight.w);
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vmins4(*this, aOther));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Vector2<T> &aOther)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vminu4(*this, aOther));
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    void Min(const Vector4<T> &aRight)
        requires HostCode<T>
    {
        x = std::min(x, aRight.x);
        y = std::min(y, aRight.y);
        z = std::min(z, aRight.z);
        w = std::min(w, aRight.w);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector4<T> &aRight)
        requires DeviceCode<T>
    {
        x = max(x, aRight.x);
        y = max(y, aRight.y);
        z = max(z, aRight.z);
        w = max(w, aRight.w);
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vmaxs4(*this, aOther));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Vector2<T> &aOther)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        *this = FromUint(__vmaxu4(*this, aOther));
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    void Max(const Vector4<T> &aRight)
        requires HostCode<T>
    {
        x = std::max(x, aRight.x);
        y = std::max(y, aRight.y);
        z = std::max(z, aRight.z);
        w = std::max(w, aRight.w);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires DeviceCode<T>
    {
        return Vector4<T>{min(aLeft.x, aRight.x), min(aLeft.y, aRight.y), min(aLeft.z, aRight.z),
                          min(aLeft.w, aRight.w)};
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        return FromUint(__vmins4(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        return FromUint(__vminu4(aLeft, aRight));
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector4<T> Min(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires HostCode<T>
    {
        return Vector4<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y), std::min(aLeft.z, aRight.z),
                          std::min(aLeft.w, aRight.w)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires DeviceCode<T>
    {
        return Vector4<T>{max(aLeft.x, aRight.x), max(aLeft.y, aRight.y), max(aLeft.z, aRight.z),
                          max(aLeft.w, aRight.w)};
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires isSameType<T, sbyte> && CUDA_ONLY<T>
    {
        return FromUint(__vmaxs4(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires isSameType<T, byte> && CUDA_ONLY<T>
    {
        return FromUint(__vmaxu4(aLeft, aRight));
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector4<T> Max(const Vector4<T> &aLeft, const Vector4<T> &aRight)
        requires HostCode<T>
    {
        return Vector4<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y), std::max(aLeft.z, aRight.z),
                          std::max(aLeft.w, aRight.w)};
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T>
    {
        return min(min(x, y), min(z, w));
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T>
    {
        return std::min({x, y, z, w});
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T>
    {
        return max(max(x, y), max(z, w));
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T>
    {
        return std::max({x, y, z, w});
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Round(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<int> RoundI(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
        ret.Round();
        return Vector4<int>(ret);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires FloatingComplexType<T>
    {
        x.Round();
        y.Round();
        z.Round();
        w.Round();
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = round(x);
        y = round(y);
        z = round(z);
        w = round(w);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    void Round()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::round(x);
        y = std::round(y);
        z = std::round(z);
        w = std::round(w);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Floor(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<int> FloorI(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
        ret.Floor();
        return Vector4<int>(ret);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE void Floor()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = floor(x);
        y = floor(y);
        z = floor(z);
        w = floor(w);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    void Floor()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::floor(x);
        y = std::floor(y);
        z = std::floor(z);
        w = std::floor(w);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> Ceil(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<int> CeilI(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
        ret.Ceil();
        return Vector4<int>(ret);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE void Ceil()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = ceil(x);
        y = ceil(y);
        z = ceil(z);
        w = ceil(w);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    void Ceil()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::ceil(x);
        y = std::ceil(y);
        z = std::ceil(z);
        w = std::ceil(w);
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> RoundNearest(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
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
        z = __float2int_rn(z);
        w = __float2int_rn(w);
    }

    /// <summary>
    /// Element wise round nearest ties to even <para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    void RoundNearest()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::nearbyint(x);
        y = std::nearbyint(y);
        z = std::nearbyint(z);
        w = std::nearbyint(w);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4<T> RoundZero(const Vector4<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4<T> ret = aValue;
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
        z = __float2int_rz(z);
        w = __float2int_rz(w);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    void RoundZero()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::trunc(x);
        y = std::trunc(y);
        z = std::trunc(z);
        w = std::trunc(w);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<T> XYZ() const
    {
        return Vector3<T>(x, y, z);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<T> YZW() const
    {
        return Vector3<T>(y, z, w);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> XY() const
    {
        return Vector2<T>(x, y);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> YZ() const
    {
        return Vector2<T>(y, z);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> ZW() const
    {
        return Vector2<T>(z, w);
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
DEVICE_CODE Vector4<T> operator+(const Vector4<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft.x + aRight), T(aLeft.y + aRight), T(aLeft.z + aRight), T(aLeft.w + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4<T> operator+(T2 aLeft, const Vector4<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft + aRight.x), T(aLeft + aRight.y), T(aLeft + aRight.z), T(aLeft + aRight.w)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4<T> operator-(const Vector4<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft.x - aRight), T(aLeft.y - aRight), T(aLeft.z - aRight), T(aLeft.w - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4<T> operator-(T2 aLeft, const Vector4<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft - aRight.x), T(aLeft - aRight.y), T(aLeft - aRight.z), T(aLeft - aRight.w)};
}

template <typename T, typename T2>
DEVICE_CODE Vector4<T> operator*(const Vector4<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft.x * aRight), T(aLeft.y * aRight), T(aLeft.z * aRight), T(aLeft.w * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4<T> operator*(T2 aLeft, const Vector4<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft * aRight.x), T(aLeft * aRight.y), T(aLeft * aRight.z), T(aLeft * aRight.w)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4<T> operator/(const Vector4<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft.x / aRight), T(aLeft.y / aRight), T(aLeft.z / aRight), T(aLeft.w / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4<T> operator/(T2 aLeft, const Vector4<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4<T>{T(aLeft / aRight.x), T(aLeft / aRight.y), T(aLeft / aRight.z), T(aLeft / aRight.w)};
}

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

} // namespace opp
